import subprocess
import time
from playwright.sync_api import sync_playwright
import pytest

@pytest.fixture(scope="module")
def start_app():
    proc = subprocess.Popen(["streamlit", "run", "app.py"])
    time.sleep(8)
    yield
    proc.terminate()

def test_home_page_header(start_app):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")
        assert page.locator("text='Medical Entity Extraction & Adverse Event Detection'").is_visible()
        browser.close()

def test_sidebar_navigation_present(start_app):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")
        assert page.locator("text='🏠 Home'").is_visible()
        assert page.locator("text='📄 Text Input & Preprocessing'").is_visible()
        assert page.locator("text='🤖 Model Training'").is_visible()
        browser.close()

def test_text_input_section(start_app):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")
        page.click("text='📄 Text Input & Preprocessing'")
        page.wait_for_selector("text='Choose input method'")
        assert page.locator("text='Manual Text Input'").is_visible()
        page.click("text='Manual Text Input'")
        page.fill("textarea", "The patient was prescribed aspirin and experienced nausea and dizziness.")
        page.click("text='Process Text'")
        page.wait_for_selector("text='✅ Text processed successfully!'")
        assert page.locator("text='✅ Text processed successfully!'").is_visible()
        browser.close()

