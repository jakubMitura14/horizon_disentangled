import pypdf
import json

def extract_annotations(pdf_path):
    """
    Extracts annotations from a PDF file and saves them to a JSON file.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    try:
        reader = pypdf.PdfReader(pdf_path)
        annotations = []
        for page_num, page in enumerate(reader.pages):
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    obj = annot.get_object()
                    if "/Contents" in obj:
                        comment = {
                            "page": page_num + 1,
                            "type": obj.get("/Subtype"),
                            "content": obj.get("/Contents")
                        }
                        annotations.append(comment)

        # Save annotations to a JSON file
        output_path = pdf_path.replace(".pdf", "_comments.json")
        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=4)

        print(f"Successfully extracted {len(annotations)} annotations to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_annotations("251014_EIC_PFC_Budget_FFB.pdf")