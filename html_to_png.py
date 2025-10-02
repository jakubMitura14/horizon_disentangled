import asyncio
from playwright.async_api import async_playwright
import sys
import os

async def html_to_png(html_file, png_file, scale_factor=2):
    """
    Converts an HTML file to a PNG image with a specified scale factor.
    A smaller scale factor (e.g., 2) creates a smaller image, which can help
    prevent "Dimension too large" errors in LaTeX.
    """
    print(f"Starting conversion of {html_file} to {png_file}...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Construct the absolute file URL correctly
        absolute_html_path = os.path.abspath(html_file)
        # The 'file://' prefix is essential for local files
        html_url = f"file://{absolute_html_path}"
        print(f"Navigating to URL: {html_url}")

        await page.goto(html_url)

        # Set a large enough viewport to capture the whole content,
        # but full_page=True should handle most cases.
        await page.set_viewport_size({"width": 1280, "height": 960})

        print(f"Taking screenshot and saving to {png_file}...")
        # Use full_page=True to capture the entire rendered HTML content
        await page.screenshot(
            path=png_file,
            type="png",
            full_page=True,
            timeout=60000  # Increased timeout for complex pages
        )
        
        await browser.close()
        print(f"Successfully converted {html_file} to {png_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python html_to_png.py <input_html_file> <output_png_file>")
        sys.exit(1)

    input_html = sys.argv[1]
    output_png = sys.argv[2]

    # Check if the input file exists
    if not os.path.exists(input_html):
        print(f"Error: Input file not found at {input_html}")
        sys.exit(1)

    # Run the conversion
    asyncio.run(html_to_png(input_html, output_png))