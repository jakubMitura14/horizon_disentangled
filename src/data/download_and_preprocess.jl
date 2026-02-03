using JSON
using CSV
using DataFrames
using NIfTI
using Printf
using Glob

# --- Configuration ---
const DATA_DIR = get(ENV, "DATA_DIR", "src/data_store")
# Local paths from user
const LOCAL_PATHS = Dict(
    "Prostate158" => "/home/jm/Downloads/6481141/prostate158_train",
    "QIN" => "/home/jm/Downloads/QIN_dataset/manifest-1557521391779/QIN-PROSTATE-Repeatability",
    "Biopsy" => "/home/jm/Downloads/prostate_mri_us/manifest-1694710246744/Prostate-MRI-US-Biopsy"
)

# --- Helper Functions ---

function symlink_dataset(source_path, link_name)
    target = joinpath(DATA_DIR, link_name)
    if ispath(target) || islink(target)
        # remove existing to update or if it's a broken link
        rm(target; force=true, recursive=true)
    end
    # Ensure parent dir exists
    mkpath(dirname(target))
    symlink(source_path, target)
    println("Symlinked $source_path -> $target")
end




function convert_dicom_to_nifti(input_dir, output_dir)
    mkpath(output_dir)
    
    # Check if NIfTIs already exist to skip
    if !isempty(glob("*.nii.gz", output_dir))
        # println("  Skipping dcm2niix (files exist) for $output_dir") 
        return true
    end

    # Uses dcm2niix
    dcm2niix_bin = strip(read(`find src/bin -name dcm2niix`, String))
    if isempty(dcm2niix_bin)
        dcm2niix_bin = "dcm2niix" # Fallback to path
    end
    
    # -z y: compress (y), -i y: ignore derived (y), -m y: merge 2D (y)
    cmd = `$dcm2niix_bin -z y -f "%p_%s" -o $output_dir $input_dir`
    try
        run(pipeline(cmd, stdout=devnull, stderr=devnull))
        return true
    catch e
        println("  dcm2niix failed for $input_dir: $e")
        return false
    end
end

function generate_dummy_mask(reference_nii_path, mask_path)
    # Load reference to get size
    nii = niread(reference_nii_path)
    dims = size(nii)
    # Create zero mask of same size
    # Just write the volume, ignoring exact header copy to avoid mutability issues
    dummy_data = zeros(UInt8, dims)
    niwrite(mask_path, NIVolume(dummy_data))
    println("    Generated dummy mask at $mask_path")
end

function identify_modalities(patient_dir, nii_files)
    t2w = nothing
    adc = nothing
    mask = nothing
    
    for f in nii_files
        fn = lowercase(basename(f))
        
        # Priority 1: Mask Detection
        # Check for explicit mask keywords FIRST before modality checks
        # "segmentation" handles QIN long filenames
        # "anatomy" handles Prostate158 whole gland
        # "tumor" handles Prostate158 tumor
        is_mask_candidate = occursin("mask", fn) || occursin("seg", fn) || 
                            occursin("anatomy", fn) || occursin("tumor", fn)
        
        if is_mask_candidate
            # If multiple masks exist, we might overwrite. 
            # Heuristic: Prefer "anatomy" (whole gland) or generic "segmentation" over specific "tumor" for general segmentation tasks if both exist?
            # Or just take the first/last found.
            # In Prostate158: t2_anatomy_reader1 vs t2_tumor_reader1. We want anatomy usually for whole gland seg used in supervisors.
            
            if mask === nothing
                mask = f
            elseif occursin("anatomy", fn)
                # Overwrite if we find an anatomy mask (preferred) and previous wasn't (e.g. was tumor)
                mask = f
            end
            
            # Continue to next file, do NOT check for t2w/adc if it is a mask
            continue
        end

        # Priority 2: Modalities
        if occursin("t2", fn)
            t2w = f
        elseif occursin("adc", fn)
            adc = f
        end
    end
    
    # Fallbacks for modalities only
    if t2w === nothing && !isempty(nii_files)
        # Scan again for any non-mask file? Or just take first?
        # Let's take first non-mask if possible
        for f in nii_files
             fn = lowercase(basename(f))
             if !(occursin("mask", fn) || occursin("seg", fn) || occursin("anatomy", fn) || occursin("tumor", fn))
                 t2w = f
                 break
             end
        end
        # If still nothing, take ANY file (desperate fallback)
        if t2w === nothing; t2w = nii_files[1]; end
    end
    
    if adc === nothing 
        adc = t2w # Use T2W as ADC if missing
    end
    
    return t2w, adc, mask
end

function process_dicom_dataset(dataset_name, source_link_name, max_cases=2)
    println("\nProcessing $dataset_name (DICOM -> NIfTI)...")
    base_dir = joinpath(DATA_DIR, source_link_name)
    output_base = joinpath(DATA_DIR, "processed", dataset_name)
    mkpath(output_base)

    patient_dirs = filter(isdir, readdir(base_dir, join=true))
    patient_dirs = filter(p -> !startswith(basename(p), "."), patient_dirs)
    targets = patient_dirs[1:min(length(patient_dirs), max_cases)]
    
    clinical_records = DataFrame(patient_id=String[], dataset=String[], t2w_path=String[], adc_path=String[], seg_path=String[])

    for p_dir in targets
        pid = basename(p_dir)
        println("  Converting patient: $pid")
        
        out_dir = joinpath(output_base, pid)
        if convert_dicom_to_nifti(p_dir, out_dir)
            nii_files = glob("*.nii.gz", out_dir)
            if !isempty(nii_files)
                t2w, adc, mask = identify_modalities(out_dir, nii_files)
                
                # Handle missing mask
                if mask === nothing && t2w !== nothing
                    mask_path = joinpath(out_dir, "dummy_mask.nii.gz")
                    generate_dummy_mask(t2w, mask_path)
                    mask = mask_path
                end
                
                if t2w !== nothing
                    println("    Identified T2W: $(basename(t2w))")
                    push!(clinical_records, (pid, dataset_name, abspath(t2w), abspath(adc), abspath(mask)))
                end
            else
                println("    WARNING: No NIfTI files created for $pid")
            end
        end
    end
    
    return clinical_records
end

function process_prostate158(source_link_name, max_cases=2)
    println("\nProcessing Prostate158 (Organizing NIfTIs)...")
    base_dir = joinpath(DATA_DIR, source_link_name, "train")
    output_base = joinpath(DATA_DIR, "processed", "Prostate158")
    mkpath(output_base)
    
    if !isdir(base_dir)
        println("  ERROR: Expected 'train' subdirectory in Prostate158 not found at $base_dir")
        return DataFrame()
    end

    patient_dirs = filter(isdir, readdir(base_dir, join=true))
    targets = patient_dirs[1:min(length(patient_dirs), max_cases)]
    
    clinical_records = DataFrame(patient_id=String[], dataset=String[], t2w_path=String[], adc_path=String[], seg_path=String[])

    for p_dir in targets
        pid = basename(p_dir)
        println("  Processing patient: $pid")
        
        out_dir = joinpath(output_base, pid)
        mkpath(out_dir)
        
        files = glob("*.nii.gz", p_dir)
        for f in files
            cp(f, joinpath(out_dir, basename(f)); force=true)
        end
        
        # In Prostate158, filenames are usually t2.nii.gz, adc.nii.gz, t2_anatomy_reader1.nii.gz (mask)
        # We need to construct absolute paths for the CSV
        files_new = glob("*.nii.gz", out_dir)
        t2w, adc, mask = identify_modalities(out_dir, files_new)
        
         if mask === nothing && t2w !== nothing
            mask_path = joinpath(out_dir, "dummy_mask.nii.gz")
            generate_dummy_mask(t2w, mask_path)
            mask = mask_path
        end
        
        if t2w !== nothing
             push!(clinical_records, (pid, "Prostate158", abspath(t2w), abspath(adc), abspath(mask)))
        end
    end
    
    return clinical_records
end


# --- Main Logic ---

function main()
    mkpath(DATA_DIR)
    println("=" ^ 60)
    println("Data Pipeline (Local) - DATA_DIR=$DATA_DIR")
    println("=" ^ 60)

    # 1. Symlink Datasets
    println("\n=== Symlinking Local Datasets ===")
    symlink_dataset(LOCAL_PATHS["Prostate158"], "raw_Prostate158")
    symlink_dataset(LOCAL_PATHS["QIN"], "raw_QIN")
    symlink_dataset(LOCAL_PATHS["Biopsy"], "raw_Biopsy")

    # 2. Process Datasets
    debug_str = get(ENV, "DEBUG_MODE", "true")
    if debug_str == "false"
        println("\nDEBUG_MODE=false: Processing LIMITED FULL DATASET (50 cases each)")
        println("  (Limiting to 50 per dataset due to disk space constraints on /media/jm/hddData)")
        max_cases = 50 
    else
        println("\nDEBUG_MODE=true: Processing subset (2 cases)")
        max_cases = 2
    end
    
    all_clinical = DataFrame()
 
    # MRI-US Biopsy (has > 400 cases, limit to 50)
    df_biopsy = process_dicom_dataset("PROSTATE-MRI-US-BIOPSY", "raw_Biopsy", max_cases)
    append!(all_clinical, df_biopsy)
 
    # QIN (small, ~15)
    df_qin = process_dicom_dataset("QIN-PROSTATE-Repeatability", "raw_QIN", max_cases)
    append!(all_clinical, df_qin)
     
    # Prostate158 (~140, limit to 50)
    df_p158 = process_prostate158("raw_Prostate158", max_cases)
    append!(all_clinical, df_p158)
    
    # 3. Save Clinical Data CSV
    println("\n=== Saving Metadata ===")
    csv_path = joinpath(DATA_DIR, "clinical_data.csv")
    CSV.write(csv_path, all_clinical)
    println("Saved cohort metadata to $csv_path")
    println(all_clinical)

    println("\n" * "=" ^ 60)
    println("Data Pipeline Complete")
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
