using NIfTI
using Statistics

function check_stats(path)
    if !isfile(path)
        println("File not found: $path")
        return
    end
    
    println("Checking $path...")
    nii = niread(path)
    img = nii.raw
    
    println("Shape: ", size(img))
    println("Type: ", eltype(img))
    
    non_zeros = count(x -> x != 0, img)
    total_voxels = length(img)
    percent = 100 * non_zeros / total_voxels
    
    println("Non-zero voxels: $non_zeros")
    println("Percent non-zero: $percent%")
    
    if non_zeros > 0
        println("Min: ", minimum(img))
        println("Max: ", maximum(img))
        println("Unique values: ", length(unique(img)))
    end
end

if length(ARGS) > 0
    check_stats(ARGS[1])
else
    # Default
    check_stats("verification_data/nifti_outputs/moose_pet_julia_output.nii.gz")
end
