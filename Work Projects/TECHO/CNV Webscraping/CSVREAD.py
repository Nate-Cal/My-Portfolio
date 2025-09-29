import csv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1

csv_path = os.path.expanduser('~/Downloads/leads.csv')

# Authenticate Google credentials
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('voluntariado-2025-2426287dcc9f.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
spreadsheet = client.open("Copia de TECHO - Arg | CNV : Gestión de datos (2025)")
worksheet = spreadsheet.worksheet("Datos recibidos")

# Get all data (including headers)
all_values = worksheet.get_all_values()

# Get headers
headers = all_values[1]
print("Headers found in sheet:", headers)

# Dynamically find column indexes
hora_col = headers.index("Horario de envío") + 1
nombre_col = headers.index("Nombre Completo") + 1
email_col = headers.index("E-mail") + 1
telefono_col = headers.index("Telefono") + 1
fecha_col = headers.index("FECHA") + 1

# Get duplicates
existing_emails = set(worksheet.col_values(email_col)[2:])

# Data containers
dates, times, emails, names, phone = [], [], [], [], []

# Parse CSV and store desired columns
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        email = row.get("Email","").strip()
        if email in existing_emails:
            print(f"Duplicate email found, skipping: {email}")
            continue

        dt_raw = row["\ufeffCreated"].strip()
        try:
            date_part, time_part = dt_raw.split(' ', 1)
            dates.append(date_part)
            times.append(time_part)
        except ValueError:
            print(f"Skipping invalid datetime: {dt_raw}")
            continue

        emails.append(email)  
        names.append(row['Name'])  
        phone.append(row['Phone'])
        print("Elements stored")


# Find next empty row (counting existing rows)
col_values = worksheet.col_values(nombre_col)
next_row = len(col_values) + 1

# Resize worksheet if needed
required_rows = next_row + len(dates) - 1
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
update_column(hora_col, times, "Horario de envío")
update_column(nombre_col, names, "Nombre Completo")
update_column(email_col, emails, "E-mail")
update_column(telefono_col, phone, "Telefono")
update_column(fecha_col, dates, "FECHA")

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