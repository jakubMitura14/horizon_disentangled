from docx import Document

data = {
    "Full Name:": " Prof. Dr. Michael Kreißl",
    "Institution:": " University Hospital Magdeburg, Germany",
    "Position/Role in the project:": " Principal Investigator (PI)",
    "Email address:": " Not provided",
    "Phone number:": " Not provided",
}

doc = Document("grant_proposal/ProstateNET_Data_Access_Request_Form.docx")

instructions = {
    "1.1. Research Team": "Jakub Mitura, Joanna Wybrańska.",
    "2. Project Title/Study Name": "Generative AI for Nuclear Medicine Optimization (CausalPCa)",
    "3. Research Question / Aim": "Leverage advanced generative architectures to synthesize high-fidelity medical imagery to enhance diagnostic precision while prioritizing patient safety, and modeling prostate cancer disease evolution with a causal AI framework.",
    "4. Scientific Background/Rationale": "The project aims to develop a novel AI framework for managing prostate cancer by moving beyond simple prediction to a deep, causal understanding of disease progression. The goal is to create a transparent decision support tool that can integrate diverse, irregularly sampled clinical data (MRI, PET/CT, SPECT/CT, PSA, notes) and provide explainable insights through counterfactual reasoning. By disentangling the underlying biological signals from technical confounders, the model will simulate disease trajectories and treatment outcomes, mirroring a clinician's own reasoning process and fostering clinical trust.",
    "5. Methodology": "Multi-stage causal AI framework, generative models (Diffusion, VAEs, Transformers), Neural Jump ODEs for temporal trajectory modeling. The framework is built upon a sequential, multi-stage training process: Training Foundational Supervisor Models, Per-Modality Causal Representation Learning, Temporal Trajectory Modeling with Neural Jump ODEs, and Generative Synthesis and Clinical Outputs.",
    "6. Dataset Requirements": "~100,000 DICOM/NIfTI files (MRI, PET/CT, SPECT/CT). Dataset: ~300 TB (Raw/Processed). Stored in HDF5 format for efficient parallel I/O. Access to Exascale capabilities is vital for training high-fidelity generative models on this large dataset. Hybrid Julia/Python stack.",
    "7. Expected Results and Applicability": "High-fidelity synthetic image generation, counterfactual disease trajectories, radiation reduction. The AI system is designed as a decision support tool for medical professionals. Final diagnostic and treatment decisions remain with human doctors. The system provides probabilistic outputs and uncertainty estimates to aid, not replace, human judgment.",
    "8. Funding Information": "Public Funding (University/Hospital)",
    "9. Study Duration": "36 Months (from May 1, 2025 to April 30, 2028).",
    "10. Ethical and Legal Considerations": "Strictly synthetic and/or fully anonymized retrospective data, no PII, complies with GDPR and local regulations. Data usage is strictly for research purposes with appropriate ethics committee approval. Data is pseudonymized at source and anonymized for processing. Access is restricted to authorized personnel.",
    "11. Supporting Documents": "PI CV uploaded."
}

def clean_insert(doc, title, new_text):
    for i, p in enumerate(doc.paragraphs):
        if p.text.startswith(title):
            # Target instruction
            target_idx = i
            while target_idx + 1 < len(doc.paragraphs):
                next_p = doc.paragraphs[target_idx+1]
                if next_p.text.strip() and not next_p.text[0].isdigit() and not next_p.style.name.startswith("Heading"):
                    target_idx += 1
                else:
                    break

            # Since the script modifies in place and can be run multiple times,
            # we need to cleanly check if it's already there
            search_idx = i + 1
            found = False
            while search_idx < len(doc.paragraphs):
                check_p = doc.paragraphs[search_idx]
                if check_p.text and check_p.text[0].isdigit() and (check_p.text[1] == '.' or (len(check_p.text) > 2 and check_p.text[2] == '.')):
                    break
                if check_p.text == new_text:
                    found = True
                    break
                search_idx += 1

            if not found:
                if target_idx + 1 < len(doc.paragraphs):
                    doc.paragraphs[target_idx + 1].insert_paragraph_before(new_text)
                else:
                    doc.add_paragraph(new_text)
            return

for p in doc.paragraphs:
    for key, val in data.items():
        if p.text.startswith(key):
            if not p.text.endswith(val):
                p.text = p.text.replace(key, key + val)

for title, text in instructions.items():
    clean_insert(doc, title, text)


doc.save("grant_proposal/ProstateNET_Data_Access_Request_Form.docx")
