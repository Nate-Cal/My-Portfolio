from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import os

# Path to the ChromeDriver

chrome_driver_path = r'C:\'
   
# Set up Chrome options

chrome_options = Options()
chrome_options.add_argument("--start-maximized") # Start browser maximized

# Optionally set download preferences

download_dir = r'C:\'
chrome_options.add_experimental_option('prefs', {
	'download.default_directory': download_dir,
	'download.prompt_for_download': False,
	'download.directory_upgrade': True,
	'safebrowsing.enabled': True
})
# Initialize the driver

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
	# Open the website

	driver.get('URL')

	# Wait for the page to load

	time.sleep(5)

	# Locate the button and click it
	
	button = driver.find_element(By.ID, 'button-id') # Adjust the locator as needed
	button.click()

	# Wait for the download to complete
	
	time.sleep(10)

finally:
	# Close the browser
	
	driver.quit()

