module SlidingWindow

using Lux
using CUDA
using Statistics

"""
    predict_sliding_window(model, x, ps, st, patch_size; overlap=0.5)

Run Leaky Sliding Window Inference on 3D input `x`.
x: (W, H, D, C, N) - usually N=1
patch_size: (pW, pH, pD)
overlap: Float, usually 0.5.
Returns: (Output, st)
Output: (W, H, D, NumClasses, N)
"""
function predict_sliding_window(model, x, ps, st, patch_size; overlap=0.5)
    W, H, D, C, N = size(x)
    pW, pH, pD = patch_size
    
    # Calculate strides
    sW = ceil(Int, pW * (1 - overlap))
    sH = ceil(Int, pH * (1 - overlap))
    sD = ceil(Int, pD * (1 - overlap))
    
    # Steps
    stepsW = 1:sW:(W - pW + sW)
    stepsH = 1:sH:(H - pH + sH)
    stepsD = 1:sD:(D - pD + sD)
    
    # We need to handle edges where patch goes out of bounds
    # Usually we clamp or padding?
    # nnU-Net tactic: if (start + patch_size) > dim, shift start back to (dim - patch_size).
    
    startsW = [min(s, W - pW + 1) for s in stepsW]
    startsH = [min(s, H - pH + 1) for s in stepsH]
    startsD = [min(s, D - pD + 1) for s in stepsD]
    
    # Make unique (in case last step overlaps exactly with shifted last patch)
    startsW = unique(startsW)
    startsH = unique(startsH)
    startsD = unique(startsD)
    
    # Initialize buffers
    # We need num_classes. Run first patch.
    first_patch = x[1:pW, 1:pH, 1:pD, :, :]
    y1, st_new = model(first_patch, ps, st)
    
    # y1 shape: (pW, pH, pD, num_classes, N)
    num_classes = size(y1, 4)
    
    # Create accumulator on CPU to save memory?
    # Or on Device if it fits?
    # 50M voxels * 84 classes * 4 bytes = 16GB.
    # Matches my RAM limit. 
    # Best to keep accumulator on CPU.
    
    output_sum = zeros(Float32, W, H, D, num_classes, N)
    count_map = zeros(Float32, W, H, D, 1, 1)
    
    total_patches = length(startsW) * length(startsH) * length(startsD)
    current_patch = 0
    
    println("Starting Sliding Window Inference. Total Patches: $total_patches")
    println("Output Shape: ", size(output_sum))
    
    for d in startsD
        for h in startsH
            for w in startsW
                current_patch += 1
                if current_patch % 10 == 0
                    print("\rPatch $current_patch / $total_patches")
                end
                
                # Extract patch
                # x might be on GPU.
                slice = x[w:w+pW-1, h:h+pH-1, d:d+pD-1, :, :]
                
                # Predict
                y, _ = model(slice, ps, st)
                
                # Move to CPU for accumulation
                y_cpu = Array(y)
                
                # Accumulate
                output_sum[w:w+pW-1, h:h+pH-1, d:d+pD-1, :, :] .+= y_cpu
                count_map[w:w+pW-1, h:h+pH-1, d:d+pD-1, :, :] .+= 1.0
            end
        end
    end
    println("\nInference Complete. Averaging...")
    
    # Normalize
    output_final = output_sum ./ count_map
    
    return output_final, st_new
end

end
