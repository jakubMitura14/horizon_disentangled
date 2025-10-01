import asyncio
import argparse
from pathlib import Path
from playwright.async_api import async_playwright

async def html_to_png(html_path: Path, png_path: Path, scale_factor: int):
    """
    Converts an HTML file to a high-quality PNG image with a specified scale factor.

    Args:
        html_path: Path to the input HTML file.
        png_path: Path to save the output PNG file.
        scale_factor: The device scale factor to use for rendering (e.g., 2 for Retina).
    """
    if not html_path.is_file():
        print(f"Error: Input file not found at {html_path}")
        return

    print(f"Starting conversion with a scale factor of {scale_factor}...")

    async with async_playwright() as p:
        # We use chromium as it's generally good for rendering.
        browser = await p.chromium.launch()
        
        # Create a new browser context with the desired device scale factor.
        # This simulates a high-DPI display.
        context = await browser.new_context(
            device_scale_factor=scale_factor
        )
        page = await context.new_page()

        # Convert the local file path to a file:// URL
        uri = html_path.resolve().as_uri()

        print(f"Navigating to {uri}...")
        # Wait until the page is fully loaded, including network resources.
        await page.goto(uri, wait_until="networkidle")

        print(f"Taking screenshot and saving to {png_path}...")
        
        # The screenshot will now be rendered at the higher resolution.
        await page.screenshot(path=str(png_path), full_page=True)

        await browser.close()
        print("Conversion complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a local HTML file to a high-quality PNG image."
    )
    parser.add_argument(
        "input_html",
        type=str,
        help="Path to the input HTML file."
    )
    parser.add_argument(
        "output_png",
        type=str,
        help="Path to save the output PNG file."
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Device scale factor for the screenshot resolution. Default is 2 (e.g., for Retina displays)."
    )
    args = parser.parse_args()

    input_path = Path(args.input_html)
    output_path = Path(args.output_png)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(html_to_png(input_path, output_path, args.scale))

if __name__ == "__main__":
    main()
