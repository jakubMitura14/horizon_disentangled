import re

def clean_bib_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # A dictionary of replacements to be made.
    replacements = {
        '&amp;': r'\\&',
        '&': r'\&',
        '<i>': r'\\textit{',
        '</i>': '}',
        '<b>': r'\\textbf{',
        '</b>': '}',
    }

    # A list of regex patterns for more complex replacements.
    regex_replacements = [
        (r'<sup>([0-9]+)</sup>', r'\\textsuperscript{\1}'),
        (r'<sup>Ⓡ</sup>', r'\\textsuperscript{\\textregistered}'),
        (r'&lt;sup&gt;(.*?)&lt;\\/sup&gt;', r'\\textsuperscript{\1}'),
        (r'&lt;sup&gt;Ⓡ&lt;\\/sup&gt;', r'\\textsuperscript{\\textregistered}'),

    ]

    # Perform the simple string replacements.
    for old, new in replacements.items():
        content = content.replace(old, new)

    # Perform the regex-based replacements.
    for pattern, repl in regex_replacements:
        content = re.sub(pattern, repl, content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    clean_bib_file("bibl.bib")
    # Add the missing citation
    with open("bibl.bib", "a", encoding="utf-8") as f:
        f.write("""
@article{AhmadzadehfarEssler2018,
  author    = {Ahmadzadehfar, Hojjat and Essler, Markus and Rahbar, Kambiz and Afshar-Oromieh, Ali},
  title     = {Radionuclide Therapy for Bone Metastases: Utility of Scintigraphy and PET Imaging for Treatment Planning},
  journal   = {PET Clinics},
  volume    = {13},
  number    = {4},
  pages     = {491--503},
  year      = {2018},
  doi       = {10.1016/j.cpet.2018.05.001}
}
""")