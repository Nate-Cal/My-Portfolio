import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import logging
from selenium.common.exceptions import (
    NoSuchElementException, 
    TimeoutException, 
    StaleElementReferenceException,
    ElementClickInterceptedException,
    WebDriverException
)
from typing import List, Dict, Optional, Tuple

# Constants
URL = "https://business.facebook.com/latest/leads_center"
COOKIE_FILE = "cookies.pkl"
OUTPUT_FILE = "leads_full_data.csv"
CSV_COLUMNS = ["full_name", "email", "phone", "dob", "province", "dni"]
MAX_ATTEMPTS = 3
WAIT_TIMEOUT = 25
DETAIL_VIEW_SELECTORS = [
    "//div[contains(@class, 'x14vqqas') and contains(@class, 'x1ey2m1c')]",  # Common dialog class
    "//div[@role='dialog']",
    "//div[contains(@aria-label, 'Lead details')]"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lead_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_driver() -> webdriver.Chrome:
    """Configure and return Chrome WebDriver with optimal settings."""
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Additional reliability options
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    
    service = webdriver.ChromeService()
    driver = webdriver.Chrome(options=options, service=service)
    driver.implicitly_wait(5)
    return driver

def load_cookies(driver: webdriver.Chrome) -> bool:
    """Load authentication cookies from file."""
    try:
        driver.get("https://facebook.com")
        time.sleep(2)  # Allow page to load
        
        with open(COOKIE_FILE, "rb") as f:
            cookies = pickle.load(f)
            
        for cookie in cookies:
            # Clean cookie dictionary
            cookie_dict = {k: v for k, v in cookie.items() 
                         if k in ['name', 'value', 'domain', 'path', 'expiry']}
            try:
                driver.add_cookie(cookie_dict)
            except Exception as e:
                logger.warning(f"Couldn't add cookie {cookie['name']}: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Cookie loading failed: {e}")
        return False

def wait_for_element(driver, by: str, value: str, timeout: int = WAIT_TIMEOUT) -> Optional[WebDriverWait]:
    """Wait for element with multiple fallback strategies."""
    try:
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
    except TimeoutException:
        logger.warning(f"Element not found: {by}={value}")
        return None

def navigate_to_leads_table(driver: webdriver.Chrome) -> bool:
    """Navigate to the leads table view."""
    try:
        # Wait for page to fully load
        wait_for_element(driver, By.TAG_NAME, "body")
        
        # Switch to table view
        table_view = wait_for_element(driver, By.XPATH, "//span[text()='Table view']")
        if table_view:
            driver.execute_script("arguments[0].click();", table_view)
        else:
            raise Exception("Table view button not found")
        
        # Select forms dropdown
        forms_dropdown = wait_for_element(driver, By.XPATH, "//div[text()='Forms']")
        if forms_dropdown:
            driver.execute_script("arguments[0].click();", forms_dropdown)
        else:
            raise Exception("Forms dropdown not found")
        
        # Select specific form
        target_form = wait_for_element(driver, By.XPATH, "//div[text()='Voluntariado 2025 - Leads-copy']")
        if target_form:
            driver.execute_script("arguments[0].click();", target_form)
        else:
            raise Exception("Target form not found")
        
        return True
    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        return False

def extract_from_detail_view(driver: webdriver.Chrome) -> Optional[Dict[str, str]]:
    """Extract data from the detailed view popup with robust selectors."""
    detail_view = None
    for selector in DETAIL_VIEW_SELECTORS:
        try:
            detail_view = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, selector))
            )
            break
        except TimeoutException:
            continue
    
    if not detail_view:
        logger.error("Detail view not found with any selector")
        return None
    
    lead_data = {}
    field_map = {
        'full name': 'full_name',
        'email address': 'email',
        'phone number': 'phone',
        'date of birth': 'dob',
        'province': 'province',
        'dni': 'dni'
    }
    
    try:
        # Find all field containers
        containers = detail_view.find_elements(By.XPATH, ".//div[contains(@class, 'x14vqqas')]")
        if not containers:
            containers = detail_view.find_elements(By.XPATH, ".//div[contains(@class, 'x1sxyh0')]")
        
        for container in containers:
            try:
                # Try multiple selector patterns for label and value
                label_elem = container.find_element(By.XPATH, ".//div[contains(@class, 'xmi5d70')]")
                value_elem = container.find_element(By.XPATH, ".//div[contains(@class, 'x117nqv4') or contains(@class, 'x1yc453h')]")
                
                label = label_elem.text.strip().lower()
                value = value_elem.text.strip()
                
                if label in field_map:
                    lead_data[field_map[label]] = value
                else:
                    logger.debug(f"Unmapped field found: {label}")
                    
            except NoSuchElementException:
                continue
        
        # Validate essential fields
        if not all(field in lead_data for field in ['full_name', 'email', 'phone']):
            logger.warning("Missing essential fields in detail view")
            return None
            
        return lead_data
        
    except Exception as e:
        logger.error(f"Detail extraction error: {e}")
        return None

def close_detail_view(driver: webdriver.Chrome) -> bool:
    """Attempt to close the detail view using multiple strategies."""
    close_selectors = [
        "//div[@aria-label='Close']",
        "//div[contains(@aria-label, 'Close')]",
        "//div[text()='Close']",
        "//button[contains(@aria-label, 'Close')]"
    ]
    
    for selector in close_selectors:
        try:
            close_btn = driver.find_element(By.XPATH, selector)
            driver.execute_script("arguments[0].click();", close_btn)
            time.sleep(0.5)
            return True
        except:
            continue
    
    # Final fallback - ESC key
    try:
        from selenium.webdriver.common.keys import Keys
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
        return True
    except:
        logger.warning("Could not close detail view")
        return False

def process_single_lead(driver: webdriver.Chrome, row, attempt: int = 1) -> Optional[Dict[str, str]]:
    """Process one lead with robust error handling."""
    if attempt > MAX_ATTEMPTS:
        logger.warning(f"Max attempts reached for lead row")
        return None
        
    try:
        # Scroll to and click the row
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", row)
        time.sleep(0.5)
        
        try:
            row.click()
        except (ElementClickInterceptedException, WebDriverException):
            driver.execute_script("arguments[0].click();", row)
        
        logger.info(f"Attempt {attempt}: Processing lead...")
        
        # Extract data
        lead_data = extract_from_detail_view(driver)
        
        # Close detail view
        close_detail_view(driver)
        
        if lead_data:
            return lead_data
        elif attempt < MAX_ATTEMPTS:
            time.sleep(1)
            return process_single_lead(driver, row, attempt + 1)
        return None
        
    except Exception as e:
        logger.error(f"Error processing lead (attempt {attempt}): {str(e)}")
        close_detail_view(driver)
        if attempt < MAX_ATTEMPTS:
            time.sleep(1)
            return process_single_lead(driver, row, attempt + 1)
        return None

def save_leads_to_csv(data: List[Dict[str, str]]) -> bool:
    """Save extracted leads to CSV file with error handling."""
    try:
        # Backup existing file if it exists
        import os
        if os.path.exists(OUTPUT_FILE):
            import shutil
            backup_name = f"{OUTPUT_FILE}.bak"
            shutil.copy2(OUTPUT_FILE, backup_name)
            logger.info(f"Created backup of existing file: {backup_name}")
        
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Successfully saved {len(data)} leads to {OUTPUT_FILE}")
        return True
    except Exception as e:
        logger.error(f"CSV save failed: {e}")
        return False

def main():
    """Main execution flow with comprehensive error handling."""
    driver = None
    try:
        driver = setup_driver()
        logger.info("Driver initialized")
        
        # Initial setup
        driver.get(URL)
        time.sleep(3)  # Initial page load
        
        if not load_cookies(driver):
            raise Exception("Cookie loading failed - please authenticate manually")
            
        # Refresh after cookies
        driver.get(URL)
        wait_for_element(driver, By.TAG_NAME, "body")
        
        # Navigation
        if not navigate_to_leads_table(driver):
            raise Exception("Navigation to leads table failed")
        
        # Wait for leads to load
        lead_rows = wait_for_element(driver, By.XPATH, "//tr[contains(@class, 'xwebqov')]", 30)
        if not lead_rows:
            raise Exception("No lead rows found")
        
        # Process all visible leads
        lead_list = []
        rows = driver.find_elements(By.XPATH, "//tr[contains(@class, 'xwebqov')]")
        logger.info(f"Found {len(rows)} leads to process")
        
        for idx, row in enumerate(rows, 1):
            logger.info(f"Processing lead {idx}/{len(rows)}...")
            lead_data = process_single_lead(driver, row)
            if lead_data:
                lead_list.append(lead_data)
                logger.debug(f"Extracted data: {lead_data}")
            else:
                logger.warning(f"Failed to process lead {idx}")
            time.sleep(0.5)  # Gentle delay between leads
        
        # Save results
        if lead_list:
            if not save_leads_to_csv(lead_list):
                raise Exception("Failed to save leads data")
        else:
            logger.warning("No leads were extracted")
            
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("Browser closed")
            except:
                pass

if __name__ == "__main__":
    main()