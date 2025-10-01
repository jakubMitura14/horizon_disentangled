import re

def remove_duplicate_bib_entries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = re.split(r'(^@)', content, flags=re.MULTILINE)

    unique_entries = {}

    # Process entries, starting from the first '@'
    i = 1
    while i < len(entries):
        # Each entry is composed of the '@' symbol and the following content
        entry_content = entries[i] + entries[i+1]

        # Extract the citation key (e.g., 'FrancoHayward2012')
        match = re.search(r'@\w+\{([^,]+),', entry_content)
        if match:
            key = match.group(1).strip()
            if key not in unique_entries:
                unique_entries[key] = entry_content
        i += 2

    # Rebuild the file content from unique entries
    new_content = "".join(unique_entries.values())

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == "__main__":
    remove_duplicate_bib_entries("bibl.bib")