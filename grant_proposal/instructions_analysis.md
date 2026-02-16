# Instructions Analysis for EuroHPC "AI for Science" Proposal

This document maps the "Instructions: Not provided" placeholders in `main_julich.tex` (derived from the administrative form PDF) to likely instructions based on EuroHPC guidelines and the "Regular Access" project scope template.

## 1. The Project

**Placeholder Location:** `\section*{The Project} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Provide a clear and concise description of the project. The summary should be suitable for publication and explain the main objectives, methodologies, and expected impact of the AI for Science project. Select keywords that best describe the scientific and technical scope.

**Action:**
*   Replace "Not provided" with a summary of the project scope instructions.
*   Ensure the "Project summary" and "Keywords" fields are filled (which they are).

## 2. Organization details

**Placeholder Location:** `\subsection*{Organization details} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Provide details of the Principal Investigator's primary affiliation. This organization will be the main contact point for administrative matters. Ensure the address and R&D figures are accurate.

**Action:**
*   Replace "Not provided" with instructions about organizational details.

## 3. Partitions

**Placeholder Location:** `\section*{Partitions} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Select the specific partition(s) on the target supercomputer (e.g., JUPITER Booster). Specify the total amount of resources requested in Node Hours or GPU Hours. Ensure the request matches the resource justification provided in the Project Scope and Plan (Part B). Note the typical job characteristics (wall clock time, checkpoints, core/GPU counts).

**Action:**
*   Replace "Not provided" with instructions on resource selection and justification.

## 4. Code details, Development and Data management

**Placeholder Location:** `\section*{Code details, Development and Data management} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Describe the technical implementation of your project.
> *   **Workflows:** Detail the AI/HPC workflows, including training, inference, and data processing steps. Mention specific frameworks (PyTorch, Julia, etc.) and parallelization strategies.
> *   **Scalability:** Provide evidence of your code's performance and scalability on the target architecture (or similar). Include strong/weak scaling results and benchmark data (e.g., from KISSKI).
> *   **Data:** Confirm if the project data and results will be open to the community.

**Action:**
*   Replace "Not provided" with instructions emphasizing technical readiness and benchmarking.

## 5. Application Support Team (AST)

**Placeholder Location:** `\section*{Application Support Team (AST)} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Indicate if your project requires dedicated support from the hosting centre's Application Support Team (AST) or Simulation Labs for code porting, optimization, or workflow development.

**Action:**
*   Replace "Not provided" with instructions about AST support availability.

## 6. Collaboration and Funding

**Placeholder Location:** `\section*{Collaboration and Funding} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Specify any existing collaborations with industry, academia, or the public sector. Indicate the funding sources supporting this project (e.g., University grants, EC projects, National funding).

**Action:**
*   Replace "Not provided" with instructions on declaring funding and partners.

## 7. Dissemination Strategy

**Placeholder Location:** `\section*{Dissemination Strategy} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Outline the plan for disseminating the results of the project. This includes scientific publications, conference presentations, open-source code releases, and public datasets.

**Action:**
*   Replace "Not provided" with instructions on dissemination expectations.

## 8. Ethics Self-Assessment

**Placeholder Location:** `\section*{Ethics Self-Assessment} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> This section is specific to the "AI for Science" call. Address the ethical guidelines for Trustworthy AI.
> *   **Human Agency:** How does the system support human decision-making?
> *   **Privacy:** How is personal data protected?
> *   **Fairness:** How is algorithmic bias mitigated?
> *   **Transparency:** How are users informed about the AI system?
> *   **Accountability:** What measures are in place for oversight and error correction?

**Action:**
*   Replace "Not provided" with instructions referencing the "Ethics Guidelines for Trustworthy AI".

## 9. Data Consent

**Placeholder Location:** `\section*{Data Consent} ... \textbf{Instructions:} Not provided`

**Inferred Instructions:**
> Please confirm your consent for the publication of project details if awarded. Acknowledge that you have read and understood the EuroHPC JU Access Policy and Terms of Reference.

**Action:**
*   Replace "Not provided" with the consent confirmation text.

---
**Summary of Next Steps:**
The `main_julich.tex` file will be updated to replace these "Instruction" placeholders with the inferred text, making the document a complete and self-explanatory proposal form representation.
