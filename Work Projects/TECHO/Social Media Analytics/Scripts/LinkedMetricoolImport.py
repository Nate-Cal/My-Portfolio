import csv
import os
import gspread
import chardet
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1
from datetime import datetime, date

csv_path = os.path.expanduser('~/TECHO/Social Media Analytics/LinkedIN/linkedin-posts_2024-08-05_2025-08-05.csv')

# Authenticate and open worksheet
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('voluntariado-2025-2426287dcc9f.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
spreadsheet = client.open("SOCIALES / DATOS ANALITICOS")
worksheet = spreadsheet.worksheet("LINKEDIN")

# Get all data (including headers)
all_values = worksheet.get_all_values()

# Get headers
headers = all_values[0]
print("Headers found in sheet:", headers)

# Dynamically find column indexes
titles_col = headers.index("TITLE OF POST") + 1
dates_col = headers.index("DATE POSTED") + 1
links_col = headers.index("LINK TO POST") + 1
likes_col = headers.index("LIKES") + 1
comments_col = headers.index("COMMENTS") + 1
shares_col = headers.index("SHARES") + 1
impressions_col = headers.index("IMPRESSIONS") + 1
engagement_col = headers.index("ENGAGEMENT") + 1
views_col = headers.index("VIEWS") + 1
viewers_col = headers.index("VIEWERS") + 1
timewatched_col = headers.index("TIME WATCHED") + 1
types_col = headers.index("TYPE OF POST") + 1

# Data containers
titles, dates, links, likes, comments, shares, impressions, engagement, views, viewers, timewatched, types = [], [], [], [], [], [], [], [], [], [], [], [], []

# Detect encoding
#with open(csv_path, 'rb') as rawfile:
#    result = chardet.detect(rawfile.read(5000))
#    detected_encoding = result['encoding']
#    print("Detected encoding:", detected_encoding)

# Parse CSV and store desired columns

with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    print("🔍 Raw CSV Headers (repr):")
    for i, field in enumerate(reader.fieldnames):
        print(f"{i}: {repr(field)}")

    # Strip any BOM or whitespace from header names
    reader.fieldnames = [f.strip().replace('\ufeff', '') for f in reader.fieldnames]
    for row in reader:

        # Skip <2025
        raw_date = row.get("Date", "").strip()
            
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

        titles.append(row.get("Title", "").strip())
        dates.append(row.get("Date", "").strip())
        links.append(row.get("URL", "").strip())
        likes.append(row.get("Likes", "").strip())
        comments.append(row.get("Comments", "").strip())
        shares.append(row.get("Shares", "").strip())
        impressions.append(row.get("Impressions", "").strip())
        engagement.append(row.get("Engagement", "").strip())
        views.append(row.get("Vid. Views", "").strip())
        viewers.append(row.get("Viewers", "").strip())
        timewatched.append(row.get("Time Watched", "").strip())
        types.append(row.get("Type", "").strip())
                
                
        print(f"📌 Stored row for: {row.get('Title', '').strip()}")

# Find next empty row (counting existing rows)
col_values = worksheet.col_values(titles_col)
next_row = len(col_values) + 1

# Resize worksheet if needed
max_rows = max(len(titles), len(dates), len(links), len(likes), len(comments), len(shares), len(impressions), len(engagement), len(views), len(viewers), len(timewatched), len(types))
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
update_column(titles_col, titles, "TITLE OF POST")
update_column(dates_col, dates, "DATE POSTED")
update_column(links_col, links, "LINK TO POST")
update_column(likes_col, likes, "LIKES")
update_column(comments_col, comments, "COMMENTS")
update_column(shares_col, shares, "SHARES")
update_column(impressions_col, impressions, "IMPRESSIONS")
update_column(engagement_col, engagement, "ENGAGEMENT")
update_column(views_col, views, "VIEWS")
update_column(viewers_col, viewers, "VIEWERS")
update_column(timewatched_col, timewatched, "TIME WATCHED")
update_column(types_col, types, "TYPE OF POST")

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
