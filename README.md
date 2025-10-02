# CausalPCa Grant Proposal

This repository contains the LaTeX source files and supporting materials for the **CausalPCa** project, a grant proposal for the EU Horizon Pathfinder program. The project aims to develop a novel Causal AI framework for longitudinal modeling of prostate cancer progression.

## Prerequisites

To compile the grant proposal documents, you will need the following software installed on your system:

*   **Python 3.x:** The scripts in this project are written in Python.
*   **A full LaTeX distribution:** This is required to compile the `.tex` files into PDFs. A comprehensive distribution like `TeX Live` is recommended. On Debian-based systems, you can install it with:
    ```bash
    sudo apt-get update && sudo apt-get install -y texlive-full
    ```
*   **Playwright Browsers:** The `html_to_png.py` script requires browser binaries to be installed by Playwright.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browser dependencies:**
    ```bash
    playwright install
    sudo apt-get update && apt-get install -y texlive-full
    sudo apt install texlive-latex-base
    ```

## Usage

### Converting HTML Figures to PNG

The project includes several figures created as HTML files for easy modification. These must be converted to PNG format before compiling the main LaTeX document. The `html_to_png.py` script automates this process.

To convert a single HTML file (e.g., `gantt.html`) to a PNG image in the `images/` directory, run:
```bash
python3 html_to_png.py gantt.html images/gantt.png
```

### Compiling the LaTeX Documents

The main grant proposal (`main_horizon.tex`) and the budget (`budget.tex`) can be compiled into PDF documents using the `compile_latex.py` script. This script handles the multiple runs of `pdflatex` and `bibtex` required to resolve all cross-references and citations correctly.

To compile both documents, simply run:
```bash
python3 compile_latex.py
```

This will generate `main_horizon.pdf` and `budget.pdf` in the root directory of the project.