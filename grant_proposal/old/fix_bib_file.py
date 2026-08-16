import re

def fix_bib_file(filepath):
    # Read the original file content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- Step 1: Remove Duplicate Entries ---
    # Split content by the '@' symbol that starts each entry, keeping the '@'
    # Use a lookbehind to keep the delimiter
    entries = re.split(r'(?=@)', content)

    unique_entries = {}

    for entry_content in entries:
        if not entry_content.strip():
            continue

        # Extract the citation key (e.g., 'FrancoHayward2012')
        match = re.search(r'@\w+\{([^,]+),', entry_content)
        if match:
            key = match.group(1).strip()
            if key not in unique_entries:
                unique_entries[key] = entry_content

    # --- Step 2: Clean the content of each unique entry ---
    cleaned_entries = []
    for key, entry_content in unique_entries.items():
        # A dictionary of replacements to be made.
        replacements = {
            '&amp;': r' \& ',
            '&': r' \& ',
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
            entry_content = entry_content.replace(old, new)

        # Perform the regex-based replacements.
        for pattern, repl in regex_replacements:
            entry_content = re.sub(pattern, repl, entry_content)

        cleaned_entries.append(entry_content)

    # --- Step 3: Rebuild the file content ---
    new_content = "".join(cleaned_entries)

    # --- Step 4: Write the cleaned content back to the file ---
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # --- Step 5: Add the missing citation ---
    with open(filepath, "a", encoding="utf-8") as f:
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

if __name__ == "__main__":
    fix_bib_file("bibl.bib")