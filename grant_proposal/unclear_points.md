# Unclear Points and Action Items for EuroHPC Proposal

## 1. CRITICAL: Incorrect Template Identification
*   **Issue:** The document you are filling (`grant_proposal/2025.12.11_EuroHPC_Development_Access-Proposal_Filled.docx`) is based on the **"TEMPLATE | Development Access - Final Report"** (Link 69 on EuroHPC website).
*   **Correction:** You are applying for the **AI for Science and Collaborative EU Projects** call (`EHPC-AIF` ID series), not generic Development Access.
*   **Action Required:**
    1.  **Do NOT submit the current .docx file.** It is the wrong template and the wrong call type.
    2.  **Use the Correct Template:** Download **`EuroHPC-AIF-for-Science-Access-PSP.docx`** (Link 68 on the EuroHPC AI for Science call page).
    3.  **Transfer Data:** Copy the content from our generated `grant_proposal/EuroHPC_AI_for_Science_Application_Data.md` into this new template.

## 2. Project Duration
*   **Constraint:** The AI for Science call explicitly states: *"The allocations are granted for **six (6) months**."* (Ref: Call Details).
*   **Current Request:** The previous draft requested 36 months.
*   **Update:** We have adjusted the application data (`EuroHPC_AI_for_Science_Application_Data.md`) to reflect an intensive **6-month training campaign** (e.g., 25,000 Node Hours).
*   **Advice:** Stick to the 6-month limit in your application. You can mention "renewable" or "follow-up Extreme Scale access" for the long-term vision.

## 3. Resource Request
*   **Allocation Size:** We have estimated **100,000 GPU-Hours** (25,000 Node Hours on JUPITER Booster). This fits the "Large Scale" nature of the AI for Science call.
*   **Justification:** The request is justified by the "Super Heavy" ResNet-152 workload and Memory-Bound requirements (75GB HBM3 usage) detailed in `experiments/scalability/memory_analysis.md`.

## 4. Administrative Details
*   **Proposal ID:** The text dump shows `EHPC-AIF-2026SC01-062`. **Use this ID.**
*   **Final Report Sections:** Since you must switch to the `PSP` (Project Scope and Plan) template, the "Final Report" sections (P142+) from the old doc will disappear. Do not worry about them.

## 5. Publications
*   **Advice:** Ensure you list *any* relevant publications or preprints in the PI Track Record section of the new template to demonstrate excellence.
