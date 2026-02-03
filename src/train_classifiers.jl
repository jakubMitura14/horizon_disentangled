# Training Script for Multi-Task Classifiers
# Predicts T-Stage, Gleason (ordinal), and PSA (regression) from imaging

using Lux
using Random
using Optimisers
using Zygote
using Statistics
using TensorBoardLogger
using Logging
using CUDA
using LuxCUDA

include("models/classifier.jl")
include("data/classifier_loader.jl")

function train_classifiers(;
    csv_path::String="dataset_encoded.csv",
    epochs::Int=50,
    batchsize::Int=4,
    lr::Float32=1f-4,
    patience::Int=5
)
    println("="^50)
    println("Multi-Task Classifier Training (Julia/Lux)")
    println("="^50)
    
    # Device
    dev = gpu_device()
    println("Device: $dev")
    
    # Data
    loader = get_classifier_loader(csv_path; batchsize=batchsize, shuffle=true)
    n_samples = length(loader.data)
    println("Dataset: $n_samples samples")
    
    # Split into train/val (80/20)
    n_val = max(1, div(n_samples, 5))
    n_train = n_samples - n_val
    
    # Model
    rng = Random.default_rng()
    model = MultiTaskClassifier(in_channels=1)
    ps, st = Lux.setup(rng, model)
    ps = ps |> dev
    st = st |> dev
    
    n_params = sum(length, Lux.parameterlength(model))
    println("Model parameters: ~$(n_params)")
    
    # Optimizer
    opt = Optimisers.Adam(lr)
    st_opt = Optimisers.setup(opt, ps)
    
    # Logger
    logger = TBLogger("logs/classifiers", min_level=Logging.Info)
    
    # Training loop
    best_loss = Inf
    patience_counter = 0
    
    println("\n--- Training ---")
    
    with_logger(logger) do
        for epoch in 1:epochs
            epoch_loss = 0.0f0
            epoch_t_loss = 0.0f0
            epoch_g_loss = 0.0f0
            epoch_psa_loss = 0.0f0
            steps = 0
            
            for (x, labels) in loader
                # Move to device
                x = x |> dev
                labels_gpu = (
                    T_label = labels.T_label,
                    Gleason_label = labels.Gleason_label,
                    PSA_target = labels.PSA_target |> dev
                )
                
                # Loss function for this batch
                function loss_fn(p, state)
                    outputs, st_new = model(x, p, state)
                    l, ld = compute_multitask_loss(outputs, labels_gpu)
                    return l, st_new
                end
                
                # Forward + Backward (following train_vae.jl pattern)
                (loss, st_new), back = Zygote.pullback(p -> loss_fn(p, st), ps)
                grads = back((1.0f0, nothing))[1]
                
                # Update
                st_opt, ps = Optimisers.update(st_opt, ps, grads)
                st = st_new
                
                epoch_loss += loss
                steps += 1
            end
            
            avg_loss = epoch_loss / steps
            
            @info "train" loss=avg_loss epoch=epoch
            println("Epoch $epoch/$epochs | Loss: $(round(avg_loss, digits=4))")
            
            # Early stopping
            if avg_loss < best_loss
                best_loss = avg_loss
                patience_counter = 0
                # TODO: Save best model
            else
                patience_counter += 1
            end
            
            if patience_counter >= patience
                println("\nEarly stopping at epoch $epoch (Best: $best_loss)")
                break
            end
        end
    end
    
    println("\nTraining complete! Best loss: $best_loss")
    return ps, st
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    csv_path = get(ENV, "CSV_PATH", "dataset_encoded.csv")
    epochs = parse(Int, get(ENV, "EPOCHS", "50"))
    batchsize = parse(Int, get(ENV, "BATCH_SIZE", "4"))
    
    train_classifiers(;
        csv_path=csv_path,
        epochs=epochs,
        batchsize=batchsize
    )
end
