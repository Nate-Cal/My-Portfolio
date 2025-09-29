import pickle
from selenium import webdriver

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

# GOTO WEBSITE
url = "https://business.facebook.com/latest/leads_center?asset_id=28624346294&nav_ref=manage_page_ap_plus_left_nav_leads_center_button"
driver.get(url)

input("Log in manually, then press Enter here to save cookies...")
pickle.dump(driver.get_cookies(), open("cookies.pkl", "wb"))
print("Cookies saved...")
driver.quit()

