import csv
import os
import gspread
import chardet
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1
from datetime import datetime, date

csv_path = os.path.expanduser('~/TECHO/Social Media Analytics/Facebook/facebook-reels_2024-08-05_2025-08-05.csv')

# Authenticate and open worksheet
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('voluntariado-2025-2426287dcc9f.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
spreadsheet = client.open("SOCIALES / DATOS ANALITICOS")
worksheet = spreadsheet.worksheet("FACEBOOK")

# Get all data (including headers)
all_values = worksheet.get_all_values()

# Get headers
headers = all_values[0]
print("Headers found in sheet:", headers)

# Dynamically find column indexes
links_col = headers.index("LINK TO REEL") + 1
content_col = headers.index("REEL TITLE") + 1
dates_col = headers.index("DATE POSTED (REEL)") + 1
videoviews_col = headers.index("REEL VIEWS") + 1
reach_col = headers.index("REACH (REEL)") + 1
likes_col = headers.index("LIKES") + 1
comments_col = headers.index("COMMENTS (REEL)") + 1
engagement_col = headers.index("ENGAGEMENT (REEL)") + 1
videoviewtime_col = headers.index("VIDEO VIEW TIME (IN SECONDS)") + 1
avgtimewatched_col = headers.index("AVERAGE TIME WATCHED (IN SECONDS)") + 1


# Data containers
links, content, dates, videoviews, reach, likes, comments, engagement, videoviewtime, avgtimewatched = [], [], [], [], [], [], [], [], [], []

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
        
        # Normalize timezone: '2025-07-08T14:39:31-03:00' → '2025-07-08T14:39:31-0300'
        #if dt_raw[-3] == ":" and (dt_raw[-6] == "+" or dt_raw[-6] == "-"):
        #    dt_raw = dt_raw[:-3] + dt_raw[-2:]
        #try:
        #    dt = datetime.strptime(dt_raw, "%Y-%m-%dT%H:%M:%S%z")
        #    date_part = dt.date().isoformat()
        #    time_part = dt.time().isoformat()
        #except Exception as e:
        #    print(f"⚠️ Skipping invalid datetime '{dt_raw}': {e}")
        #    continue
        
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

        links.append(row.get("Reel Link", "").strip())
        content.append(row.get("Content", "").strip())
        dates.append(row.get("Date", "").strip())
        #dates.append(date_part)
        videoviews.append(row.get("Video Views", "").strip())
        reach.append(row.get("Reach", "").strip())
        likes.append(row.get("Likes", "").strip())
        comments.append(row.get("Comments", "").strip())
        engagement.append(row.get("Engagement", "").strip())
        videoviewtime.append(row.get("Video View time (Seconds)"))
        avgtimewatched.append(row.get("Avg. time watched (Seconds)"))
        
        print(f"Stored row for {content}")
        #print("Elements stored")

# Find next empty row (counting existing rows)
col_values = worksheet.col_values(content_col)
next_row = len(col_values) + 1

# Resize worksheet if needed
max_rows = max(len(links), len(content), len(dates), len(videoviews), len(reach), len(likes), len(comments), len(engagement), len(videoviewtime), len(avgtimewatched))
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
update_column(content_col, content, "REEL TITLE")
update_column(dates_col, dates, "DATE POSTED (REEL)")
update_column(videoviews_col, videoviews, "REEL VIEWS")
update_column(reach_col, reach, "REACH (REEL)")
update_column(likes_col, likes, "LIKES")
update_column(comments_col, comments, "COMMENTS (REEL)")
update_column(engagement_col, engagement, "ENGAGEMENT (REEL)")
update_column(videoviewtime_col, videoviewtime, "VIDEO VIEW TIME (IN SECONDS)")
update_column(avgtimewatched_col, avgtimewatched, "AVERAGE TIME WATCHED (IN SECONDS)")

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
