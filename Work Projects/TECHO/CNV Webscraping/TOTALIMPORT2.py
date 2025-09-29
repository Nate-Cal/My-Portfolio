import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import logging
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lead_extraction.log'),
        logging.StreamHandler()
    ]
)

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    return driver

def load_cookies(driver, cookie_file):
    try:
        cookies = pickle.load(open(cookie_file, "rb"))
        for cookie in cookies:
            cookie.pop('samesite', None)
            driver.add_cookie(cookie)
        return True
    except Exception as e:
        logging.error(f"Failed to load cookies: {e}")
        return False

def navigate_to_leads(driver, url):
    try:
        driver.get(url)
        time.sleep(3)  # Initial page load
        
        # Switch to table view
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Table view']"))
        ).click()
        logging.info("Switched to table view")
        
        # Select forms dropdown
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//div[text()='Forms']"))
        ).click()
        
        # Select specific form
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//div[text()='Voluntariado 2025 - Leads-copy']"))
        ).click()
        
        return True
    except Exception as e:
        logging.error(f"Navigation failed: {e}")
        return False

def extract_lead_data(driver):
    try:
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
                logging.warning(f"Error extracting field: {e}")
                continue
        
        return lead_data
    except Exception as e:
        logging.error(f"Lead data extraction failed: {e}")
        return None

def save_to_csv(data, filename):
    columns = ["full name", "email", "phone number", "date of birth", "province", "dni"]
    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for lead in data:
                writer.writerow({col: lead.get(col, "") for col in columns})
        logging.info(f"Successfully saved {len(data)} leads to {filename}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")

def main():
    driver = setup_driver()
    url = "https://business.facebook.com/latest/leads_center?asset_id=28624346294&nav_ref=manage_page_ap_plus_left_nav_leads_center_button"
    
    try:
        # Initial page load
        driver.get(url)
        time.sleep(3)
        
        # Load cookies and refresh
        if not load_cookies(driver, "cookies.pkl"):
            return
        
        driver.get(url)
        time.sleep(5)
        
        # Navigation
        if not navigate_to_leads(driver, url):
            return
        
        # Wait for leads to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//tr[@role='row' and contains(@class, 'xwebqov')]"))
        )
        
        lead_list = []
        visited = set()
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                lead_rows = driver.find_elements(By.XPATH, "//tr[@role='row' and contains(@class, 'xwebqov')]")
                
                for row in lead_rows:
                    try:
                        row_id = row.text.strip()[:50]  # Truncate for logging
                        if row_id and row_id not in visited:
                            driver.execute_script("arguments[0].scrollIntoView(true);", row)
                            driver.execute_script("arguments[0].click();", row)
                            logging.info(f"Processing lead: {row_id}...")
                            
                            # Wait for lead details to load
                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.XPATH, "//div[@class='x14vqqas']"))
                            
                            lead_data = extract_lead_data(driver)
                            if lead_data:
                                lead_list.append(lead_data)
                                visited.add(row_id)
                            
                            # Go back
                            WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, "//div[@aria-label='Back' or @aria-label='Close']"))
                            ).click()
                            
                            time.sleep(1)  # Brief pause for stability
                    except StaleElementReferenceException:
                        logging.warning("Stale element reference, retrying...")
                        continue
                    except Exception as e:
                        logging.warning(f"Skipping lead due to error: {e}")
                        continue
                
                break  # Successfully processed all leads
                
            except Exception as e:
                attempt += 1
                logging.error(f"Attempt {attempt} failed: {e}")
                time.sleep(5)
                if attempt == max_attempts:
                    logging.error("Max attempts reached, aborting.")
                    break
        
        # Save results
        if lead_list:
            save_to_csv(lead_list, "leads_full_data.csv")
        else:
            logging.warning("No leads were extracted")
            
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        driver.quit()
        logging.info("Browser closed")

if __name__ == "__main__":
    main()