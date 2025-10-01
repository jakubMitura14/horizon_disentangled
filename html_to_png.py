import asyncio
from playwright.async_api import async_playwright
import sys

async def html_to_png(html_file, png_file, scale_factor=2):
    """
    Converts an HTML file to a PNG image with a specified scale factor.
    A smaller scale factor (e.g., 2) creates a smaller image, which can help
    prevent "Dimension too large" errors in LaTeX.
    """
    print(f"Starting conversion with a scale factor of {scale_factor}...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Construct the file URL
        html_path = f"file:///app/{html_file}"
        print(f"Navigating to {html_path}...")

        await page.goto(html_path)

        # Set the viewport to a reasonable size for a slide/figure
        await page.set_viewport_size({"width": 1200, "height": 800})

        print(f"Taking screenshot and saving to {png_file}...")
        await page.screenshot(path=png_file, type="png", clip=None, full_page=True, scale="device", timeout=60000)
        
        await browser.close()
        print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python html_to_png.py <input_html_file> <output_png_file>")
        sys.exit(1)

    html_file = sys.argv[1]
    png_file = sys.argv[2]

    # Run the conversion
    asyncio.run(html_to_png(html_file, png_file))