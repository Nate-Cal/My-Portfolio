from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time


chrome_options = Options()
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)
# GOTO Website
url = "https://app.metricool.com/evolution/facebookPage?blogId=2460321&userId=2083682"
driver.get(url)

print("The Email is comunicaciones.argentina@techo.org")
print("The Password is MetricoolTecho24")
input("Log in manually, then press Enter to continue...")

time.sleep(5)

# Download Post CSV
try:

    first_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'m-headline')][contains(.,'Lista de publicaciones')]//button[.//span[contains(text(),'Descargar CSV')]]"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", first_button)
    driver.execute_script("arguments[0].click();", first_button)
    print("Posts CSV button found. Downloading...")

    time.sleep(2)

    second_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'m-headline')][contains(.,'Lista de reels')]//button[.//span[contains(text(),'Descargar CSV')]]"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", second_button)
    driver.execute_script("arguments[0].click();", second_button)
    print("Reels CSV button found. Downloading...")
except Exception as e:
    print("Could not download files:", e)


# Download Reel CSV

# Move to Instagram Page
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'instagram')]"))
    )
    print("Instagram button found. Clicking...")
    button.click()
except Exception as e:
    print("Could not click 'Instagram' button:", e)

time.sleep(5)

# Move to LinkedIn Page
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'linkedin')]"))
    )
    print("LinkedIn button found. Clicking...")
    button.click()
except Exception as e:
    print("Could not click 'LinkedIn' button:", e)

time.sleep(5)

# Move to TikTok Page
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'tiktok')]"))
    )
    print("TikTok button found. Clicking...")
    button.click()
except Exception as e:
    print("Could not click 'TikTok' button:", e)

time.sleep(5)

driver.quit()