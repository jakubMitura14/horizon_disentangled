using Lux
using Random
using Optimisers
using Zygote
using Statistics
using TensorBoardLogger
using Logging
using CUDA
using LuxCUDA

include("models/vae.jl")

include("data/loader.jl")

function train()
    # Get DATA_DIR from Env
    data_dir = get(ENV, "DATA_DIR", "src/mock_data")
    loader = get_data_loader(data_dir, batchsize=1, shuffle=true)

    # Logger
    logger = TBLogger("logs/vae", min_level=Logging.Info)

    # Device: GPU if available
    dev = gpu_device() 
    println("Using device: $dev")
    
    # Model
    model = CausalVAE(16, 4)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    ps = ps |> dev
    st = st |> dev

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps)

    # Loss
    function loss_fn(p, x, m, st)
        out, st_new = model((x, m), p, st)
        recon, mu_p, log_p, mu_s, log_s = out
        
        # Reconstruction loss (MSE)
        recon_loss = mean(abs2, recon .- x)

        # KL Divergence (Standard Normal prior)
        kl_p = -0.5f0 * mean(1 .+ log_p .- abs2.(mu_p) .- exp.(log_p))
        kl_s = -0.5f0 * mean(1 .+ log_s .- abs2.(mu_s) .- exp.(log_s))
        
        loss = recon_loss + 0.1f0 * (kl_p + kl_s)
        return loss, st_new
    end

    println("--- Training Causal VAE (Lux) ---")
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
            
            for (x, m) in loader
                # Reshape to 5D (W, H, D, C, B)
                if ndims(x) == 4
                    x = reshape(x, size(x,1), size(x,2), size(x,3), 1, size(x,4))
                end
                if ndims(m) == 4
                    m = reshape(m, size(m,1), size(m,2), size(m,3), 1, size(m,4))
                end
                
                x = x |> dev
                m = m |> dev
                
                (l, st_new), back = Zygote.pullback(p -> loss_fn(p, x, m, st), ps)
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

    println("VAE Trained.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
