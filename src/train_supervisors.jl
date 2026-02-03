using Lux
using Random
using Optimisers
using Zygote
using CSV
using DataFrames
using NIfTI
using Statistics
using Printf
using TensorBoardLogger
using Logging
using CUDA
using LuxCUDA
using MLUtils # For DataLoader logic inside loader

include("models/supervisors.jl")
include("data/loader.jl")

function train()
    # Get DATA_DIR from Env
    data_dir = get(ENV, "DATA_DIR", "src/mock_data")
    loader = get_data_loader(data_dir, batchsize=1, shuffle=true)

    # Logger
    logger = TBLogger("logs/supervisors", min_level=Logging.Info)

    # Device Setup (GPU if available)
    dev = gpu_device()
    println("Using device: $dev")

    # Model
    model = UnetSupervisor(2, 3) # 3 classes
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    ps = ps |> dev
    st = st |> dev

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps)

    # Loss
    function loss_function(p, x, y, st)
        pred, st_new = model(x, p, st)
        l = mean(abs2, pred .- y)
        return l, st_new
    end

    println("--- Training Segmentation Supervisor (Lux) ---")
    println("Dataset size: $(length(loader.data)) patients")
    
    # Early Stopping Config
    max_epochs = 50
    patience = 3
    best_loss = Inf
    patience_counter = 0

    with_logger(logger) do
        for i in 1:max_epochs
            epoch_loss = 0.0f0
            steps = 0
            
            for (x, seg) in loader
                # Move to device
                x = x |> dev
                seg = seg |> dev
                
                (l, st_new), back = Zygote.pullback(p -> loss_function(p, x, seg, st), ps)
                grads = back((1.0f0, nothing))[1]

                st_opt, ps = Optimisers.update(st_opt, ps, grads)
                st = st_new
                
                epoch_loss += l
                steps += 1
            end
            
            avg_loss = epoch_loss / steps
            @info "train" loss=avg_loss epoch=i
            println("Epoch $i Avg Loss: $avg_loss")
            
            # Early Stopping
            if avg_loss < best_loss
                best_loss = avg_loss
                patience_counter = 0
            else
                patience_counter += 1
            end
            
            if patience_counter >= patience
                println("Early stopping at epoch $i (Best: $best_loss)")
                break
            end
        end
    end

    # Save checkpoint (mock)
    println("Supervisor trained.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

