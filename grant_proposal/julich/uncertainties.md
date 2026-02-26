# Uncertainties and Assumptions

## Template Compliance
1.  **Scaling Table Columns**: The GCS template explicitly requests a "table with the timings" and "parallel efficiency". The initial proposal draft lacked these specific columns (providing only Throughput and Speedup). *Action taken: Added Time (s) and Parallel Efficiency columns to the proposal.*
2.  **Resource Table Headers**: The GCS template provides example headers for CPU and GPU tables. The GPU table example in the template text uses `#host cores/run` and `Total [core-h]`.
    *   *Uncertainty*: "Total [core-h]" for a GPU request is ambiguous. It likely refers to "GPU-hours" or "Node-hours" depending on the center's accounting.
    *   *Assumption*: Used `Total [GPU-h]` as it is the most logical unit for a GPU grant.
    *   *Deviation*: Added `# GPUs/run` column which was not in the template example but is critical for clarity.
3.  **Bibliography**: The template requests "Recent/most important bibliographic references". The generated proposal includes a bibliography file `references.bib` with a vast number of entries, but the specific citations in the text are limited. *Action: Ensure relevant citations are actually keyed in the text.*

## Target System Mismatch
1.  **JUPITER vs Hunter/SuperMUC-NG**: The user provided Fact Sheets for `Hunter` and `SuperMUC-NG` (GCS centers HLRS and LRZ), but the proposal content (derived from `main_julich.tex`) targets `JUPITER` (FZJ).
    *   *Risk*: Specific constraints for JUPITER (e.g., maximum node allocation, partitions) might differ from Hunter/SuperMUC.
    *   *Assumption*: The GCS template is generic enough for JUPITER (as implied by the template text listing JUPITER), but the specific constraints in the provided PDF Fact Sheets do not apply to JUPITER.
2.  **Call ID**: The ID `EHPC-AIF-2026SC01-062` suggests an EuroHPC "AI for Science" call. The GCS template provided might be for a national GCS call. There is a potential template mismatch if the user intends to submit to EuroHPC using a GCS template.

## Technical Ambiguity
1.  **Scaling Type**: The provided scalability report describes the experiment as "Strong Scaling" but the data (constant time per epoch, increasing throughput) suggests "Weak Scaling" (fixed workload per GPU).
    *   *Action*: The proposal labels it as "Scaling behavior" to remain neutral, but presents Speedup based on Throughput.

## Administrative Information
1.  **Missing Fields**: Several PI details (Date of Birth, Phone, Nationality) were marked "Not provided" in the source `main_julich.tex`. These are likely mandatory for the final submission form but are omitted here.
