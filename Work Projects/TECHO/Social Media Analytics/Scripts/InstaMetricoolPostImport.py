import csv
import os
import gspread
import chardet
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1
from datetime import datetime, date

csv_path = os.path.expanduser('~/TECHO/Social Media Analytics/Instagram/instagram-posts_2024-08-05_2025-08-05.csv')

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
types_col = headers.index("TYPE OF POST") + 1
links_col = headers.index("LINK TO POST") + 1
content_col = headers.index("TITLE (POST)") + 1
dates_col = headers.index("DATE POSTED (POST)") + 1
views_col = headers.index("VIEWS (POST)") + 1
reach_col = headers.index("REACH (POST)") + 1
likes_col = headers.index("LIKES (POST)") + 1
saves_col = headers.index("SAVES (POST)") + 1
comments_col = headers.index("COMMENTS (POST)") + 1
shares_col = headers.index("SHARES (POST)") + 1
interactions_col = headers.index("INTERACTIONS (POST)") + 1
engagement_col = headers.index("ENGAGEMENT (POST)") + 1

# Data containers
types, links, content, dates, views, reach, likes, saves, comments, shares, interactions, engagement = [], [], [], [], [], [], [], [], [], [], [], []

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
        raw_date = row.get("Timestamp", "").strip()
    
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

        types.append(row.get("type", "").strip())
        links.append(row.get("URL", "").strip())
        content.append(row.get("Content", "").strip())
        dates.append(row.get("Timestamp", "").strip())
        views.append(row.get("Views (Organic)", "").strip())
        reach.append(row.get("Reach (Organic)", "").strip())
        likes.append(row.get("Likes", "").strip())
        saves.append(row.get("Saved", "").strip())
        comments.append(row.get("Comments", "").strip())
        shares.append(row.get("Shares", "").strip())
        interactions.append(row.get("Interactions", "").strip())
        engagement.append(row.get("Engagement", "").strip())
        
        
        print(f"Stored row for {content}")
        #print("Elements stored")

# Find next empty row (counting existing rows)
col_values = worksheet.col_values(content_col)
next_row = len(col_values) + 1

# Resize worksheet if needed
max_rows = max(len(types), len(links), len(content), len(dates), len(views), len(reach), len(likes), len(saves), len(comments), len(shares), len(interactions), len(engagement))
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

update_column(types_col, types, "TYPE OF POST")
update_column(links_col, links, "LINK TO POST")
update_column(content_col, content, "TITLE (POST)")
update_column(dates_col, dates, "DATE POSTED (POST)")
update_column(views_col, views, "VIEWS (POST)")
update_column(reach_col, reach, "REACH (POST)")
update_column(likes_col, likes, "LIKES (POST)")
update_column(saves_col, saves, "SAVES (POST)")
update_column(comments_col, comments, "COMMENTS (POST)")
update_column(interactions_col, interactions, "INTERACTIONS (POST)")
update_column(engagement_col, engagement, "ENGAGEMENT (POST)")

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
