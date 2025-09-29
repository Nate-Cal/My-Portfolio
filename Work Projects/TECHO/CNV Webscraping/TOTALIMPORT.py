import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import csv

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

time.sleep(3)

# Select Campaign
try:
    button = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//div[text()='Voluntariado 2025 - Leads-copy']"))
    )
    print("Voluntariado 2025 - Leads-copy button found. Clicking...")
    button.click()
except Exception as e:
    print("Could not click 'Voluntariado 2025 - Lead-copy' button:", e)

#DEBUGGING
#print("Dumping page HTML...")
#with open("page_dump.html", "w", encoding="utf-8") as f:
#    f.write(driver.page_source)
#print("Saved page_dump.html for inspection.")

try: 
    initial_rows = WebDriverWait(driver, 20).until(
    EC.presence_of_all_elements_located((By.XPATH, "//tr[@role='row' and contains(@class, 'xwebqov')]"))
    )
    print(f"Found {len(initial_rows)} lead rows.")
except Exception as e:
    print("Could not find lead rows:", e)
    driver.quit()
    exit()

time.sleep(3)

lead_list = []
num_leads = len(initial_rows)
lead_counter = 0
visited = set()

while True:
    try:
        lead_rows = driver.find_elements(By.XPATH, "//tr[@role='row' and contains(@class, 'xwebqov')]")


        row_to_click = None
        for row in lead_rows:
            row_id = row.text.strip()
            if row_id and row_id not in visited:
                row_to_click = row
                visited.add(row_id)
                break
        if not row_to_click:
            print("All leads visited.")
            break

        driver.execute_script("arguments[0].scrollIntoView(true);", row_to_click)
        time.sleep(1)

        driver.execute_script("arguments[0].click();", row_to_click)
        print(f"Clicked lead: {row_id[:50]}... ")

        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//div[@class='x14vqqas']"))
        )

        # Grab all lead field blocks
        fields = driver.find_elements(By.XPATH, "//div[@class='x14vqqas']")
        lead_data = {}


        for field in fields:
            try:
                key_element = field.find_element(By.XPATH, "./div[1]")
                value_element = field.find_element(By.XPATH, ".//div[2]/div")
                key = key_element.text.strip().lower()
                value = value_element.text.strip()
                lead_data[key] = value
            except Exception as e:
                print("Error extracting field:", e)
                continue

       
        # Add last lead 
        if lead_data:
            lead_list.append(lead_data)
            lead_counter += 1
        
        try:
            back_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@aria-label='Back' or @aria-label='Close']"))
            )
            back_button.click()
            time.sleep(1)
        except:
            print("Could not find back/close button, continuing...")
    
    except Exception as e:
        print(f"Skipping lead #{lead_counter+1} due to error:", e)
        continue
    
# Define CSV column order
columns = ["full name", "email", "phone number", "date of birth", "province", "dni"]

# Write to CSV
with open("leads_full_data.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()
    for lead in lead_list:
        writer.writerow({col: lead.get(col, "") for col in columns})

print(f"Extracted {len(lead_list)} leads to 'leads_full_data.csv'")

input("Press Enter to close the browser...")
driver.quit()