using JSON
using ZipFile
using CSV
using DataFrames
using NIfTI
using Printf
using Downloads

# --- Configuration ---
const DATA_DIR = get(ENV, "DATA_DIR", "src/data_store")
const DEBUG_MODE = get(ENV, "DEBUG_MODE", "true") == "true"

const TCIA_BASE_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA"

# --- Helper Functions ---

function download_with_curl(url, output_path; timeout_sec=120)
    println("Downloading: $url")
    println("  -> $output_path")
    try
        # Use Downloads.jl which wraps libcurl - more reliable than HTTP.jl for large files
        Downloads.download(url, output_path; timeout=timeout_sec)
        filesize_mb = round(filesize(output_path) / 1024 / 1024, digits=2)
        println("  Downloaded: $(filesize_mb) MB")
        return true
    catch e
        println("  Download failed: $e")
        return false
    end
end

function query_tcia_series(collection_id; timeout_sec=60)
    series_url = "$TCIA_BASE_URL/query/getSeries?Collection=$collection_id&format=json"
    println("Querying TCIA: $collection_id")
    
    try
        # Download JSON to temp file then parse
        tmp = tempname() * ".json"
        Downloads.download(series_url, tmp; timeout=timeout_sec)
        json_str = read(tmp, String)
        rm(tmp)
        series_list = JSON.parse(json_str)
        println("  Found $(length(series_list)) series")
        return series_list
    catch e
        println("  Query failed: $e")
        return nothing
    end
end

function download_tcia_series(collection_id, output_dir; max_series=1)
    mkpath(output_dir)
    
    series_list = query_tcia_series(collection_id; timeout_sec=60)
    if series_list === nothing || isempty(series_list)
        return false
    end
    
    target_series = series_list[1:min(max_series, length(series_list))]
    success_count = 0
    
    for s in target_series
        uid = s["SeriesInstanceUID"]
        println("Downloading series: $uid")
        download_url = "$TCIA_BASE_URL/query/getImage?SeriesInstanceUID=$uid"
        zip_path = joinpath(output_dir, "$(uid).zip")
        
        if download_with_curl(download_url, zip_path; timeout_sec=300)
            success_count += 1
        end
    end
    
    return success_count > 0
end

function generate_mock_data()
    println("Generating mock training data...")
    include(joinpath(@__DIR__, "mock_data.jl"))
    Base.invokelatest(generate_longitudinal_dataset, DATA_DIR; num_patients=4, max_timepoints=2)
    println("Mock data generated in $DATA_DIR")
end

# --- Main Logic ---

function main()
    mkpath(DATA_DIR)
    println("=" ^ 60)
    println("Data Pipeline - DATA_DIR=$DATA_DIR")
    println("=" ^ 60)

    # Try TCIA downloads (real data)
    println("\n=== Downloading from TCIA (1 case per dataset) ===\n")
    
    tcia_dir = joinpath(DATA_DIR, "TCIA")
    mkpath(tcia_dir)
    
    # Download from QIN (smaller, faster)
    qin_dir = joinpath(tcia_dir, "QIN")
    qin_ok = download_tcia_series("QIN-PROSTATE-Repeatability", qin_dir; max_series=1)
    
    # Download from Biopsy collection  
    biopsy_dir = joinpath(tcia_dir, "Biopsy")
    biopsy_ok = download_tcia_series("PROSTATE-MRI-US-BIOPSY", biopsy_dir; max_series=1)
    
    if qin_ok || biopsy_ok
        println("\n=== TCIA Download Successful ===")
        println("Downloaded DICOM data to: $tcia_dir")
        
        # Still generate clinical_data.csv structure for training scripts
        println("\nGenerating training data structure...")
        generate_mock_data()
    else
        println("\n=== TCIA Download Failed - Using Mock Data ===")
        generate_mock_data()
    end
    
    println("\n" * "=" ^ 60)
    println("Data Pipeline Complete")
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
