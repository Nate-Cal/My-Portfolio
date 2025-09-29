from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# --- Configure Chrome options for stability on macOS M1/M2 ---
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-extensions")

# --- Set default download directory ---
download_path = os.path.expanduser("~/TECHO/Social Media Analytics/MetricoolCSVs")
os.makedirs(download_path, exist_ok=True)
prefs = {
    "download.default_directory": download_path,
    "download.prompt_for_download": False,
    "directory_upgrade": True
}
chrome_options.add_experimental_option("prefs", prefs)

# --- Start Chrome with automatic driver ---
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# --- Open Metricool ---
url = "https://app.metricool.com/evolution/facebookPage?blogId=2460321&userId=2083682"
driver.get(url)

# --- Manual login ---
print("Log in manually in the browser window, then press Enter here...")
input("Press Enter after logging in...")

time.sleep(3)  # wait a bit to ensure dashboard loads

# --- Function to click CSV buttons by headline ---
def download_csv(headline_text):
    try:
        button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((
                By.XPATH,
                f"//div[contains(@class,'m-headline')][contains(.,'{headline_text}')]//button[.//span[contains(text(),'Descargar CSV')]]"
            ))
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
        time.sleep(0.3)  # allow any animations to finish
        driver.execute_script("arguments[0].click();", button)
        print(f"'{headline_text}' CSV button clicked. Download should start.")
        time.sleep(3)  # wait for download to initialize
    except Exception as e:
        print(f"Could not click '{headline_text}' CSV button:", e)

# --- Download Posts CSV ---
download_csv("Lista de publicaciones")

# --- Download Reels CSV ---
download_csv("Lista de reels")

# --- Optional: Navigate to Instagram, LinkedIn, TikTok pages ---
for platform in ["instagram", "linkedin", "tiktok"]:
    try:
        button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, f"//a[contains(@href, '{platform}')]"))
        )
        print(f"{platform.capitalize()} button found. Clicking...")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
        time.sleep(0.3)
        button.click()
        time.sleep(3)
    except Exception as e:
        print(f"Could not click '{platform}' button:", e)

# --- Close browser ---
driver.quit()
print(f"All tasks complete. CSV files should be in: {download_path}")
