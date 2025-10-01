import subprocess
import os

def compile_latex(file_path):
    """
    Compiles a LaTeX file to PDF using pdflatex.

    Args:
        file_path (str): The path to the .tex file to compile.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Get the directory and filename from the path
    directory, filename = os.path.split(file_path)
    if not directory:
        directory = "."

    # The base name of the file without the extension
    base_name = os.path.splitext(filename)[0]

    # We need to run pdflatex multiple times to resolve references
    for i in range(3):
        print(f"Running pdflatex on {filename} (pass {i+1})...")
        process = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", directory, filename],
            capture_output=True,
            encoding='latin-1',
            errors='ignore'
        )

        if process.returncode != 0:
            print(f"Error compiling {filename} on pass {i+1}.")
            print(process.stdout)
            print(process.stderr)
            return

    # Run bibtex to resolve citations
    print(f"Running bibtex on {base_name}...")
    process = subprocess.run(
        ["bibtex", os.path.join(directory, base_name)],
        capture_output=True,
        text=True
    )
    if process.returncode != 0:
        print(f"Error running bibtex on {base_name}.")
        print(process.stdout)
        print(process.stderr)
        # It's not always a fatal error, so we continue

    # Run pdflatex two more times to get citations correct
    for i in range(2):
        print(f"Running pdflatex on {filename} (pass {i+4} for citations)...")
        process = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", directory, filename],
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            print(f"Error compiling {filename} on pass {i+4}.")
            print(process.stdout)
            print(process.stderr)
            return

    print(f"Successfully compiled {filename} to {os.path.join(directory, base_name)}.pdf")

if __name__ == "__main__":
    compile_latex("main_horizon.tex")
    compile_latex("budget_standalone.tex")