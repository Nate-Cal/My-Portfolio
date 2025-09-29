import pickle 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

url = "https://business.facebook.com/latest/leads_center?asset_id=28624346294&nav_ref=manage_page_ap_plus_left_nav_leads_center_button"
driver.get(url)
time.sleep(3)

cookies = pickle.load(open("cookies.pkl", "rb"))
for cookie in cookies:
    cookie.pop('samesite', None)
    driver.add_cookie(cookie)

driver.get(url)

time.sleep(5)

# CHANGING EVERYTHING BELOW
# NEW PATH:
    # ALL TOOLS
    # INSTANT FORMS
    # VOLUNTARIADO 2025 - LEADS COPY
    # DOWNLOAD
    # DOWNLOAD BY DATE RAGE
    # DOWNLOAD
    # DOWNLOAD CSV LINK

# Move to table view
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//span[text()='Table view']"))
    )
    print("Table view button found. Clicking...")
    button.click()
except Exception as e:
    print("Could not click 'Table view' button:", e)

# Locate the button (Forms Dropdown)
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//div[text()='Forms']"))
    )
    print("Forms button found. Clicking...")
    button.click()
except Exception as e:
    print("Could not click 'Forms' button:", e)

# Select Campaign
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//div[text()='Voluntariado 2025 - Leads-copy']"))
    )
    print("Voluntariado 2025 - Leads-copy button found. Clicking...")
    button.click()
except Exception as e:
    print("Could not click 'Voluntariado 2025 - Lead-copy' button:", e)

# Download csv file
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//div[@role='button'][.//div[text()='Download CSV']]"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
    driver.execute_script("arguments[0].click();", button)
    print("Download CSV button found. Clicking...")
except Exception as e:
    print("Could not click 'Download CSV' button:", e)

input("Press Enter to close the browser...")
driver.quit()