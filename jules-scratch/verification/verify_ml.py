
import os
from playwright.sync_api import sync_playwright

def verify_ml_html():
    """
    Navigates to the local ml.html file and takes a screenshot.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Construct the file path to be absolute
        file_path = os.path.abspath('ml.html')

        # Use file:// protocol to open the local HTML file
        page.goto(f'file://{file_path}')

        # Take a screenshot
        page.screenshot(path='jules-scratch/verification/ml_verification.png')

        browser.close()

if __name__ == "__main__":
    verify_ml_html()
