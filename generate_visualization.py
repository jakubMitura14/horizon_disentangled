import nbformat
import pandas as pd

def extract_dataframes_from_notebook(notebook_path):
    """
    Reads a Jupyter Notebook and executes all code cells to define the
    allocation dataframes and return them.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # A namespace to hold the variables from the notebook execution
    namespace = {'pd': pd}

    # Execute all code cells in order
    for cell in nb.cells:
        if cell.cell_type == 'code':
            try:
                exec(cell.source, namespace)
            except Exception as e:
                print(f"Error executing cell: {cell.source}")
                raise e

    # Extract the required dataframes from the namespace
    try:
        df_y1 = namespace['df_alloc_y1']
        df_y2 = namespace['df_alloc_y2']
        df_y3 = namespace['df_alloc_y3']
    except KeyError as e:
        raise KeyError(f"Could not find dataframe {e} in the notebook's namespace.")

    return df_y1, df_y2, df_y3

def generate_html_report(df_y1, df_y2, df_y3, output_path):
    """
    Generates a styled HTML report from the allocation dataframes.
    """

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Project Allocation Plan</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                margin: 40px;
                background-color: #f8f9fa;
                color: #212529;
                line-height: 1.6;
            }}
            h1, h2 {{
                color: #0A369D;
                border-bottom: 2px solid #dee2e6;
                padding-bottom: 10px;
                margin-top: 40px;
            }}
            h1 {{
                text-align: center;
                font-size: 2.5em;
            }}
            .table-container {{
                margin-bottom: 40px;
                overflow-x: auto;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
                border-radius: 8px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: white;
            }}
            th, td {{
                border: 1px solid #dee2e6;
                padding: 12px 15px;
            }}
            th {{
                background-color: #4472CA;
                color: white;
                font-weight: 600;
                text-align: center;
            }}
            .role-header {{
                text-align: left;
                font-weight: bold;
                background-color: #f8f9fa;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #e9ecef;
            }}
            .non-zero {{
                font-weight: bold;
                color: #198754;
                text-align: center;
                display: block;
            }}
            .zero {{
                color: #adb5bd;
                text-align: center;
                display: block;
            }}
        </style>
    </head>
    <body>
        <h1>CausalPCa Grant Proposal: Person-Month Allocation</h1>

        <div class="table-container">
            <h2>Year 1 Allocation</h2>
            {table_y1}
        </div>

        <div class="table-container">
            <h2>Year 2 Allocation</h2>
            {table_y2}
        </div>

        <div class="table-container">
            <h2>Year 3 Allocation</h2>
            {table_y3}
        </div>

    </body>
    </html>
    """

    def dataframe_to_styled_html(df):
        # Applies a CSS class based on the cell's value for styling
        def val_formatter(val):
            if val > 0:
                # Use 'g' for general format to remove trailing zeros
                return f'<span class="non-zero">{val:g}</span>'
            else:
                return f'<span class="zero">{val:g}</span>'

        formatters = {col: val_formatter for col in df.columns}

        # Convert dataframe to HTML, escaping False to render the span tags
        html = df.to_html(classes='data-table', border=0, formatters=formatters, escape=False)
        # Apply the role-header class to the first column (index)
        html = html.replace('<th>Role</th>', '<th class="role-header">Role</th>')
        return html

    # The index name from pandas is 'Role' which we want to keep
    table_y1_html = dataframe_to_styled_html(df_y1.rename_axis('Role'))
    table_y2_html = dataframe_to_styled_html(df_y2.rename_axis('Role'))
    table_y3_html = dataframe_to_styled_html(df_y3.rename_axis('Role'))

    final_html = html_template.format(table_y1=table_y1_html, table_y2=table_y2_html, table_y3=table_y3_html)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"HTML report generated at {output_path}")

if __name__ == "__main__":
    notebook_file = 'project_analysis.ipynb'
    output_html_file = 'allocation_visualization.html'

    try:
        df1, df2, df3 = extract_dataframes_from_notebook(notebook_file)
        generate_html_report(df1, df2, df3, output_html_file)
    except Exception as e:
        print(f"An error occurred: {e}")
