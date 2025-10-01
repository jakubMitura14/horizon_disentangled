import pypdf
import sys

def read_pdf_text(filepath):
    try:
        with open(filepath, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python read_pdf.py <input_pdf> <output_txt>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2]

    file_content = read_pdf_text(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)

    print(f"Successfully extracted text from {pdf_path} to {output_path}")