import re
import os

def extract_cited_keys(tex_file):
    """Extracts all citation keys from a .tex file."""
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {tex_file} not found.")
        return set()

    # Regex to find \cite{...} commands, including multiple citations in one command
    raw_keys = re.findall(r'\\cite\{([^}]+)\}', content)
    cited_keys = set()
    for key_group in raw_keys:
        keys = [k.strip() for k in key_group.split(',')]
        cited_keys.update(keys)
    return cited_keys

def filter_bib_file(bib_file, out_file, cited_keys):
    """Filters a .bib file to include only the cited keys."""
    try:
        with open(bib_file, 'r', encoding='utf-8', errors='ignore') as f:
            bib_content = f.read()
    except FileNotFoundError:
        print(f"Error: {bib_file} not found.")
        return

    entries = bib_content.split('@')

    with open(out_file, 'w', encoding='utf-8') as f_out:
        for entry in entries:
            if not entry.strip():
                continue

            match = re.search(r'\{([^,]+),', entry)
            if match:
                key = match.group(1).strip()
                if key in cited_keys:
                    f_out.write('@' + entry)

if __name__ == "__main__":
    tex_file = 'main_horizon.tex'
    bib_file = 'bibl.bib'
    reduced_bib_file = 'bibl_reduced.bib'

    print(f"Extracting cited keys from {tex_file}...")
    cited_keys = extract_cited_keys(tex_file)

    if cited_keys:
        print(f"Found {len(cited_keys)} unique citations.")
        print(f"Filtering {bib_file} to create {reduced_bib_file}...")
        filter_bib_file(bib_file, reduced_bib_file, cited_keys)

        # Replace the original bibl.bib with the reduced one
        os.rename(reduced_bib_file, bib_file)
        print(f"Successfully updated {bib_file} with only cited references.")
    else:
        print("No citations found in the .tex file. No changes made to bibl.bib.")