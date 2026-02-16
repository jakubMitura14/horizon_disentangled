# Unclear Points and Missing Information for EuroHPC Proposal (main_julich.tex)

The following information is missing or marked as "Not provided" in the current draft of the proposal and requires your input to finalize the submission.

## Administrative Information
*   **PI Date of Birth:** Currently marked as "Not provided". This is a mandatory field.
*   **PI Phone Number:** Currently marked as "Not provided". This is a mandatory field.
*   **Secondary Email:** Currently set to `jakub.mitura@ovgu.de` (PI: Michael Kreißl). Confirm if this is the correct secondary contact or if it should be the PI's secondary email.
*   **Team Member Participation:** Section regarding participation in other EC actions (ERC, Marie Curie) is marked "Not provided". Please confirm if any team members are involved in such grants.
*   **Previous Presentations:** Information on previous EuroHPC/PRACE presentations is "Not provided". If applicable, list dates/titles.

## Technical Specifications
*   **CPU Cores per Job/Node:** The fields for "Maximum # CPU cores per job", "Average # CPU cores per job", and "# of CPU cores used per node" are currently "Not provided".
    *   *Recommendation:* Since we are using JUPITER Booster (Grace-Hopper), we should likely specify the number of Grace CPU cores corresponding to the 4 GPUs requested per node (e.g., 72 cores per GPU or full node 288 cores). Please confirm the desired CPU allocation strategy.
*   **Storage Quotas:** The request lists 200TB for Scratch/Archive but only 1TB for HOME and 50TB for WORK. Please confirm these quotas align with the specific policies of the JUPITER system (some centers have stricter limits on HOME/WORK).

## Reviewers
*   **Excluded Reviewers:** Currently no reviewers are excluded. If there are specific competitors or conflicts of interest you wish to avoid, please provide their Names, Emails, and Affiliations.

## Resource Estimation
*   **Node Hours Validation:** The request is for **25,000 Node Hours** (approx. 100,000 GPU Hours).
    *   *Action:* Please verify this total against your specific project budget or the maximum allowed for this call (Regular vs. Development Access).
    *   *Note:* The text mentions a "monthly plan" but only provides percentages (10%, 30%, 60%). A more detailed breakdown (e.g., "Month 1-3: 2,000 hours for profiling") might be required by reviewers.

## Policy & Ethics
*   **Dual Use:** The proposal is marked as "Civilian purposes: Yes". Confirm there are no dual-use implications (military/defense) given the high-resolution generative capabilities.
*   **Data Management:** Confirm that the 200TB dataset is indeed fully anonymized and that you have the necessary ethics approval (as stated in the text) for transferring it to the JUPITER supercomputer.
