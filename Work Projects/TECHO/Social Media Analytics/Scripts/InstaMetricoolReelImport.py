import csv
import os
import gspread
import chardet
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1
from datetime import datetime, date

csv_path = os.path.expanduser('~/TECHO/Social Media Analytics/Instagram/instagram-reels_2024-08-05_2025-08-05.csv')

# Authenticate and open worksheet
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('voluntariado-2025-2426287dcc9f.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
spreadsheet = client.open("SOCIALES / DATOS ANALITICOS")
worksheet = spreadsheet.worksheet("INSTAGRAM")

# Get all data (including headers)
all_values = worksheet.get_all_values()

# Get headers
headers = all_values[0]
print("Headers found in sheet:", headers)

# Dynamically find column indexes
links_col = headers.index("LINK TO REEL") + 1
titles_col = headers.index("TITLE (REEL)") + 1
dates_col = headers.index("DATE POSTED (REEL)") + 1
reach_col = headers.index("REACH (REEL)") + 1
likes_col = headers.index("LIKES (REEL)") + 1
saves_col = headers.index("SAVES (REEL)") + 1
comments_col = headers.index("COMMENTS (REEL)") + 1
shares_col = headers.index("SHARES (REEL)") + 1
interactions_col = headers.index("INTERACTIONS (REEL)") + 1
engagement_col = headers.index("ENGAGEMENT (REEL)") + 1

# Data containers
links, titles, dates, reach, likes, saves, comments, shares, interactions, engagement = [], [], [], [], [], [], [], [], [], []

# Detect encoding
with open(csv_path, 'rb') as rawfile:
    result = chardet.detect(rawfile.read(5000))
    detected_encoding = result['encoding']
    print("Detected encoding:", detected_encoding)

# Parse CSV and store desired columns
with open(csv_path, newline='', encoding=detected_encoding) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    print("🔍 Raw CSV Headers (repr):")
    for i, field in enumerate(reader.fieldnames):
        print(f"{i}: {repr(field)}")

    # Strip any BOM or whitespace from header names
    reader.fieldnames = [f.strip().replace('\ufeff', '') for f in reader.fieldnames]
    for row in reader:
        
        # Skip <2025
        raw_date = row.get("date", "").strip()
    
        # Skip if no date
        if not raw_date:
            continue

        try:
            post_date = datetime.strptime(raw_date, "%Y-%m-%d %H:%M")  # Adjust format if needed
        except ValueError:
            print(f"⚠️ Invalid date format skipped: {raw_date}")
            continue

        if post_date.year < 2025:
            continue  # Skip this row
        
        links.append(row.get("URL", "").strip())
        titles.append(row.get("title", "").strip())
        dates.append(row.get("date", "").strip())
        reach.append(row.get("Reach (Organic)", "").strip())
        likes.append(row.get("Likes (Organic)", "").strip())
        saves.append(row.get("Saved (Organic)", "").strip())
        comments.append(row.get("Comments (Organic)", "").strip())
        shares.append(row.get("Shares (Organic)", "").strip())
        interactions.append(row.get("Interactions (Organic)", "").strip())
        engagement.append(row.get("Engagement (Organic)", "").strip())
        

        
        print(f"Stored row for {titles}")
        #print("Elements stored")

# Find next empty row (counting existing rows)
col_values = worksheet.col_values(titles_col)
next_row = len(col_values) + 1

# Resize worksheet if needed
max_rows = max(len(links), len(titles), len(dates),  len(views), len(reach), len(likes), len(saves), len(comments), len(shares), len(interactions), len(engagement))
required_rows = next_row + max_rows - 1
if required_rows > worksheet.row_count:
    worksheet.add_rows(required_rows - worksheet.row_count)
    print(f"Worksheet resized: added {required_rows - worksheet.row_count} rows")

# Prepare columns to batch-update
def update_column(col_index, values, label):
    start_cell = rowcol_to_a1(next_row, col_index)
    end_cell = rowcol_to_a1(next_row + len(values) - 1, col_index)
    range_str = f'{start_cell}:{end_cell}'
    column_data = [[v] for v in values]
    worksheet.update(range_str, column_data)
    print(f"✅ Updated {label} column at {range_str}")

# Batch update columns


update_column(links_col, links, "LINK TO REEL")
update_column(titles_col, titles, "TITLE (REEL)")
update_column(dates_col, dates, "DATE POSTED")
update_column(reach_col, reach, "REACH (REEL)")
update_column(likes_col, likes, "LIKES (REEL)")
update_column(saves_col, saves, "SAVES (REEL)")
update_column(comments_col, comments, "COMMENTS (REEL)")
update_column(shares_col, shares, "SHARES (REEL)")
update_column(interactions_col, interactions, "INTERACTIONS (REEL)")
update_column(engagement_col, engagement, "ENGAGEMENT (REEL)")


print("Data Imported Correctly!")

# Optional CSV deletion
if os.path.exists(csv_path):
    confirm = input(f"Do you want to delete the CSV file at {csv_path}? (y/n): ").strip().lower()
    if confirm == 'y':
        os.remove(csv_path)
        print(f"CSV file deleted: {csv_path}")
    else:
        print("CSV file NOT deleted.")
else:
    print(f"File not found: {csv_path}")
