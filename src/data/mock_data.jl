using NIfTI
using CSV
using DataFrames
using Random
using Printf

function create_mock_volume(shape=(48, 48, 16); noise_level=0.1, tumor_present=false, scanner_bias=0.0)
    vol = rand(Float32, shape...) .* Float32(noise_level) .+ Float32(scanner_bias)
    mask = zeros(Float32, shape...)

    # Create Prostate Blob
    cx, cy, cz = shape .÷ 2
    r_prostate = minimum(shape) ÷ 4

    # Grid
    x = range(1, shape[1], length=shape[1])
    y = range(1, shape[2], length=shape[2])
    z = range(1, shape[3], length=shape[3])

    for k in 1:shape[3], j in 1:shape[2], i in 1:shape[1]
        if (i - cx)^2 + (j - cy)^2 + (k - cz)^2 <= r_prostate^2
            vol[i, j, k] += 0.5
            mask[i, j, k] = 1.0
        end
    end

    # Create Tumor Blob (if present)
    if tumor_present
        tx, ty, tz = cx + 5, cy + 5, cz
        r_tumor = r_prostate ÷ 3
        for k in 1:shape[3], j in 1:shape[2], i in 1:shape[1]
            if (i - tx)^2 + (j - ty)^2 + (k - tz)^2 <= r_tumor^2
                vol[i, j, k] -= 0.2
                mask[i, j, k] = 2.0 # Tumor label
            end
        end
    end

    return vol, mask
end

function generate_longitudinal_dataset(output_dir; num_patients=10, max_timepoints=3)
    mkpath(joinpath(output_dir, "images"))

    clinical_data = DataFrame(
        patient_id = String[],
        timepoint_id = String[],
        time_months = Float64[],
        age = Float64[],
        psa = Float64[],
        gleason = Int[],
        genetic_risk = Int[],
        scanner_type = String[],
        biopsy_performed = Int[],
        biopsy_coords = String[],
        t2w_path = String[],
        adc_path = String[],
        seg_path = String[],
        event_occurred = Int[],
        time_to_event = Float64[]
    )

    for i in 1:num_patients
        patient_id = @sprintf("Patient-%03d", i-1)
        age_base = rand(50:80)
        genetic_risk = rand() < 0.2 ? 1 : 0
        scanner_type = rand(["Siemens", "Philips"])
        scanner_bias = scanner_type == "Philips" ? 0.1 : 0.0

        num_visits = rand(1:max_timepoints)

        for t in 0:(num_visits-1)
            timepoint_id = "$(patient_id)_T$t"
            time_months = Float64(t * 6)

            # Disease progression logic
            tumor_present = (t > 0) || (rand() > 0.5)
            gleason = tumor_present ? rand([6, 7, 8, 9]) : 0
            psa = tumor_present ? 2.0 + (t * 1.5) : 2.0 + rand()

            # Biopsy simulation
            biopsy_performed = (t > 0 && rand() > 0.6)
            biopsy_coords = ""
            if biopsy_performed
                # Normalized coordinates 0-1 approx
                bx, by, bz = 0.5 + randn()*0.05, 0.5 + randn()*0.05, 0.5 + randn()*0.05
                biopsy_coords = "$bx,$by,$bz"
            end

            # Generate Images (48x48x16 for speed)
            t2w, seg = create_mock_volume(tumor_present=tumor_present, scanner_bias=scanner_bias)
            adc, _ = create_mock_volume(tumor_present=tumor_present, noise_level=0.2, scanner_bias=scanner_bias)

            t2w_path = joinpath(output_dir, "images", "$(timepoint_id)_t2w.nii.gz")
            adc_path = joinpath(output_dir, "images", "$(timepoint_id)_adc.nii.gz")
            seg_path = joinpath(output_dir, "images", "$(timepoint_id)_seg.nii.gz")

            # Save NIfTI
            niwrite(t2w_path, NIVolume(t2w))
            niwrite(adc_path, NIVolume(adc))
            niwrite(seg_path, NIVolume(seg))

            push!(clinical_data, (
                patient_id, timepoint_id, time_months,
                age_base + (time_months/12.0), psa, gleason, genetic_risk,
                scanner_type, biopsy_performed ? 1 : 0, biopsy_coords,
                t2w_path, adc_path, seg_path,
                (gleason >= 8 ? 1 : 0), 24.0
            ))
        end
    end

    CSV.write(joinpath(output_dir, "clinical_data.csv"), clinical_data)
    println("Generated longitudinal data for $num_patients patients in $output_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_longitudinal_dataset("src/mock_data")
end
