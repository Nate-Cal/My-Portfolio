import csv
import os
import gspread
import chardet
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import rowcol_to_a1
from datetime import datetime, date

csv_path = os.path.expanduser('~/TECHO/CSVFILES/Voluntariado 2025 - Leads- con ciudad_Leads_2025-07-08_2025-07-13.csv')

# Authenticate and open worksheet
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('voluntariado-2025-2426287dcc9f.json', scope)
client = gspread.authorize(creds)

# Open the Google Sheet
spreadsheet = client.open("TECHO - Arg | CNV : Gestión de datos (2025)")
worksheet = spreadsheet.worksheet("Datos recibidos")

# Get all data (including headers)
all_values = worksheet.get_all_values()

# Get headers
headers = all_values[1]
print("Headers found in sheet:", headers)

# Dynamically find column indexes
hora_col = headers.index("Horario de envío") + 1
nombre_col = headers.index("Nombre Completo") + 1
edad_col = headers.index("Edad") + 1
dni_col = headers.index("DNI") + 1
email_col = headers.index("E-mail") + 1
telefono_col = headers.index("Telefono") + 1
province_col = headers.index("Provincia") + 1
fecha_col = headers.index("FECHA") + 1

# Get duplicates
# NOT EMAIL
# existing_emails = set(email.lower() for email in worksheet.col_values(email_col)[2:])
existing_dnis = set(worksheet.col_values(dni_col)[2:])

unlisted_provinces = [
    'mexico',
    'san juan',
    'san luis',
]

# Data containers
dates, times, names, emails, phones, ages, province_list, dni_list = [], [], [], [], [], [], [], []

# Detect encoding
with open(csv_path, 'rb') as rawfile:
    result = chardet.detect(rawfile.read(5000))
    detected_encoding = result['encoding']
    print("Detected encoding:", detected_encoding)

# Parse CSV and store desired columns
with open(csv_path, newline='', encoding=detected_encoding) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    print("🔍 Raw CSV Headers (repr):")
    for i, field in enumerate(reader.fieldnames):
        print(f"{i}: {repr(field)}")
    
    # Force detection of correct fieldname for created_time
    created_time_key = next((f for f in reader.fieldnames if "created_time" in f.lower()), None)
    if not created_time_key:
        raise Exception("❌ Could not detect the 'created_time' field in headers")

    print(f"✅ Detected 'created_time' header as: {repr(created_time_key)}")

    # Strip any BOM or whitespace from header names
    reader.fieldnames = [f.strip().replace('\ufeff', '') for f in reader.fieldnames]
    for row in reader:
        
        # Remove invalid provinces 
        province_value = row.get("province", "").strip().lower()
        if province_value in unlisted_provinces:
            print(f"TECHO not present in {province_value}, skipping...")
            continue
        
        # Check for duplicates
        dni_value = row.get("dni", "").strip().lower()
        if dni_value in existing_dnis:
            print(f"Duplicate entry found, skipping: {dni_value}...")
            continue
        dt_raw = row.get(created_time_key, "").strip()
        if not dt_raw:
            print("⚠️ Skipping row with missing created_time...")
            continue
        
        # Fix time formatting for import
        # Normalize timezone: '2025-07-08T14:39:31-03:00' → '2025-07-08T14:39:31-0300'
        if dt_raw[-3] == ":" and (dt_raw[-6] == "+" or dt_raw[-6] == "-"):
            dt_raw = dt_raw[:-3] + dt_raw[-2:]
        try:
            dt = datetime.strptime(dt_raw, "%Y-%m-%dT%H:%M:%S%z")
            date_part = dt.date().isoformat()
            time_part = dt.time().isoformat()
        except Exception as e:
            print(f"⚠️ Skipping invalid datetime '{dt_raw}': {e}")
            continue
        
        dates.append(date_part)
        times.append(time_part)
        names.append(row.get("full_name", "").strip())
        emails.append(row.get("email", "").strip())
        phones.append(row.get("phone_number", "").strip())
        # Calculate Age
        dob_str = row.get("date_of_birth", "").strip()
        if dob_str:
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                today = date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                ages.append(age)
            except ValueError:
                print(f"Invalid date_of_birth format '{dob_str}', skipping age calculation...")
                ages.append(dob_str)
        else:
            ages.append(dob_str)
        province_list.append(row.get("ciudad", "").strip())
        dni_list.append(dni_value)
        print(f"Stored row for {dni_value}")
        #print("Elements stored")

# Find next empty row (counting existing rows)
col_values = worksheet.col_values(nombre_col)
next_row = len(col_values) + 1

# Resize worksheet if needed
max_rows = max(len(names), len(ages), len(dni_list), len(emails), len(phones), len(province_list))
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
update_column(hora_col, times, "Horario de envío")
update_column(nombre_col, names, "Nombre Completo")
update_column(edad_col, ages, "Edad")
update_column(dni_col, dni_list, "DNI")
update_column(email_col, emails, "E-mail")
update_column(telefono_col, phones, "Telefono")
update_column(province_col, province_list, "Provincia")
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
