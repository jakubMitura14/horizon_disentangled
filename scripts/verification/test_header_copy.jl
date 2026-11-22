using NIfTI

function test_copy()
    ref_path = "verification_data/nifti_outputs/moose_pet_python_output.nii.gz"
    out_path = "verification_data/nifti_outputs/test_header_copy.nii.gz"
    
    println("Reading $ref_path...")
    ref = niread(ref_path)
    
    println("Ref Header: ", ref.header)
    
    # Create dummy data same size
    data = similar(ref.raw)
    fill!(data, 1)
    
    # Try creating new NIVolume with Ref header
    # Attempt 1: Just overwrite raw?
    # ref.raw = data 
    # niwrite...
    # But this modifies ref in memory.
    
    println("Creating new NIVolume...")
    # NIVolume(header, extensions, raw)
    new_ni = NIVolume(ref.header, ref.extensions, data)
    
    println("Writing $out_path...")
    niwrite(out_path, new_ni)
    println("Done.")
end

test_copy()
