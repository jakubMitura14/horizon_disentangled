import re

def clean_bib_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace HTML ampersand
    content = content.replace('&amp;', r'\&')
    content = content.replace('&', r'\&')

    # Replace sup tags for numbers, e.g., <sup>223</sup>
    content = re.sub(r'<sup>([0-9]+)</sup>', r'\\textsuperscript{\1}', content)

    # Replace sup tags for registered trademark, e.g., <sup>Ⓡ</sup>
    content = content.replace('<sup>Ⓡ</sup>', r'\textsuperscript{\textregistered}')

    # Replace i and b tags
    content = content.replace('<i>', r'\textit{')
    content = content.replace('</i>', '}')
    content = content.replace('<b>', r'\textbf{')
    content = content.replace('</b>', '}')

    # Handle partially decoded HTML tags from previous attempts
    content = re.sub(r'&lt;sup&gt;(.*?)&lt;\\/sup&gt;', r'\\textsuperscript{\1}', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    clean_bib_file("bibl.bib")