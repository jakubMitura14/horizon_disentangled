import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Screenshot gantt.html
        gantt_path = os.path.abspath('gantt.html')
        await page.goto(f'file://{gantt_path}')
        await page.screenshot(path='jules-scratch/verification/gantt.png')

        # Screenshot ml.html
        ml_path = os.path.abspath('ml.html')
        await page.goto(f'file://{ml_path}')
        await page.screenshot(path='jules-scratch/verification/ml.png')

        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
