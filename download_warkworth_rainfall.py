import requests
import datetime
import pandas as pd
import io # To read CSV string/bytes into pandas
import json
import re # For parsing filename from header

# --- Configuration ---
BASE_URL = "https://environmentauckland.org.nz"
TOKEN_ENDPOINT = BASE_URL + "/Export/BulkExportToken"
DATA_ENDPOINT_BASE = BASE_URL + "/Export/BulkExport"
EXPORT_PAGE_URL = BASE_URL + "/Data/Export" # Used as Referer

# Known IDs (Confirmed from your data)
LOCATION_ID = "238" # Warkworth Sewage Treatment Plant
DATASET_ID = "16234" # Rainfall.Continuous@644626
DATASET_UNIT_ID = "332" # Unit ID for Rainfall (mm?) - from payload

# --- !! VERY IMPORTANT !! Session and Antiforgery Values ---
# These WILL expire. Replace with fresh values from your browser's
# Developer Tools -> Network Tab -> Copy as cURL on the POST request -> extract cookies
# or implement dynamic fetching for reliable automation.
SESSION_COOKIE_VALUE = "PLACEHOLDER_SESSION_COOKIE_VALUE" # e.g., CfDJ8O%2FXyd...
ANTIFORGERY_COOKIE_VALUE = "PLACEHOLDER_ANTIFORGERY_COOKIE_VALUE" # e.g., CfDJ8O_Xyd...
ANTIFORGERY_TOKEN_PAYLOAD_VALUE = "PLACEHOLDER_ANTIFORGERY_TOKEN_VALUE_FROM_PAYLOAD" # e.g., CfDJ8O_Xyd...

# --- Calculate Dates for last 6 hours (UTC as TimeZone=0 was used) ---
# Using timezone-aware UTC datetime objects
end_time_dt_utc = datetime.datetime.now(datetime.timezone.utc)
start_time_dt_utc = end_time_dt_utc - datetime.timedelta(hours=6)
# Format as YYYY-MM-DD HH:MM
start_time_str = start_time_dt_utc.strftime('%Y-%m-%d %H:%M')
end_time_str = end_time_dt_utc.strftime('%Y-%m-%d %H:%M')
print(f"Requesting data from {start_time_str} UTC to {end_time_str} UTC")

# --- Prepare Session and Headers ---
session = requests.Session()
headers = {
    'User-Agent': 'Python Automation Script (EnvironmentAuckland Download)',
    'Accept': '*/*', # Default for token request
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': BASE_URL,
    'Referer': EXPORT_PAGE_URL,
    'DNT': '1',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'empty', # Default for token request
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'TE': 'trailers',
}

# Set cookies MANUALLY in the session object (REPLACE WITH YOUR VALUES)
session.cookies.set('.AspNetCore.Session', SESSION_COOKIE_VALUE, domain='environmentauckland.org.nz', path='/')
session.cookies.set('.AspNetCore.Antiforgery.MsBUFd0jSQI', ANTIFORGERY_COOKIE_VALUE, domain='environmentauckland.org.nz', path='/')
session.cookies.set('disclaimer', 'accepted', domain='environmentauckland.org.nz', path='/')
session.cookies.set('userutcoffset', '-600', domain='environmentauckland.org.nz', path='/') # From your example capture
session.cookies.set('i18n.langtag', 'en', domain='environmentauckland.org.nz', path='/')

# --- Prepare POST Payload (Using settings from your working 7-day example) ---
payload_dict = {
    'StartTime': start_time_str, # Calculated start for last 6h
    'EndTime': end_time_str,   # Calculated end for last 6h
    'exportSelect': '',
    'DateRange': 'Custom', # Set to Custom since we provide Start/End
    'TimeZone': '0', # UTC
    'Calendar': 'CALENDARYEAR',
    # *** Use settings from the payload you confirmed worked ***
    'Interval': 'PointsAsRecorded',
    'Step': '1',
    'ExportFormat': 'csv',
    'TimeAligned': 'True', # Keep True for this test, as bulk export defaults likely expect it
    'RoundData': 'False',
    'IncludeGradeCodes': 'False',
    'IncludeInterpolationTypes': 'False',
    'IncludeQualifiers': 'False',
    f'Datasets[0].ColumnOrder': '0',
    f'Datasets[0].LocationId': LOCATION_ID,
    f'Datasets[0].LocationName': f'{LOCATION_ID} - Mahurangi @ Warkworth Sewage Treatment Plant',
    f'Datasets[0].DatasetId': DATASET_ID,
    f'Datasets[0].DatasetName': f'Rainfall.Continuous@{LOCATION_ID}',
    # *** Use settings from the payload you confirmed worked ***
    f'Datasets[0].Calculation': 'Instantaneous',
    f'Datasets[0].UnitId': DATASET_UNIT_ID,
    'accept': 'true',
    '__RequestVerificationToken': ANTIFORGERY_TOKEN_PAYLOAD_VALUE # *** USER MUST REPLACE ***
}
print(f"Using Interval: {payload_dict['Interval']}, Calculation: {payload_dict[f'Datasets[0].Calculation']}")

# --- Step 1: POST to get Token ---
try:
    print(f"Sending POST to get token: {TOKEN_ENDPOINT}")
    token_headers = headers.copy()
    token_headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'

    token_response = session.post(TOKEN_ENDPOINT, headers=token_headers, data=payload_dict)
    token_response.raise_for_status()
    print(f"Token request status: {token_response.status_code}")

    response_json = token_response.json()
    download_token = response_json.get('Token')

    if not download_token:
        print("Error: Token not found in response.")
        print("Response:", token_response.text)
        exit()

    print("Received download token.")

    # --- Step 2: POST to get Data using Token ---
    data_endpoint = f"{DATA_ENDPOINT_BASE}?Token={download_token}"
    print(f"Sending POST to get data: {data_endpoint}")

    data_headers = headers.copy()
    data_headers['Content-Type'] = 'application/x-www-form-urlencoded'
    data_headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    data_headers['Sec-Fetch-Dest'] = 'iframe'
    data_headers['Sec-Fetch-Mode'] = 'navigate'

    data_response = session.post(data_endpoint, headers=data_headers, data=payload_dict)
    data_response.raise_for_status()
    print(f"Data request status: {data_response.status_code}")

    # --- Process Data Response ---
    if 'text/csv' in data_response.headers.get('Content-Type', '').lower():
        print("CSV data received.")

        raw_csv_text = data_response.text
        print("\n--- Raw CSV Text (first ~10 lines) ---")
        lines = raw_csv_text.splitlines()
        for i, line in enumerate(lines):
            if i >= 10:
                break
            print(f"L{i}: {line}")
        print("-------------------------------------")

        print("Loading into DataFrame...")
        csv_data = io.StringIO(raw_csv_text)

        try:
            # Using header=3, comment='#' based on previous successful header parsing
            df = pd.read_csv(csv_data, header=3, comment='#')

            print("\n--- Downloaded DataFrame (first 5 rows) ---")
            print(df.head())
            print("\n--- DataFrame Info ---")
            df.info()

            if df.empty:
                print("\n*** DataFrame is empty. ***")
                print("Even with 'PointsAsRecorded', no data was returned for this period.")
                print("Verify manually on the website if data exists for this exact UTC time range.")

        except pd.errors.EmptyDataError:
            print("\nError: The downloaded CSV file was empty after skipping comments/header row.")
        except Exception as e:
            print(f"\nError parsing CSV data into DataFrame: {e}")

    else:
        print(f"Error: Expected CSV data, but received Content-Type: {data_response.headers.get('Content-Type')}")
        print("Response Text:", data_response.text[:500] + "...")

except requests.exceptions.RequestException as e:
    print(f"\nError during network request: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print("Response text:", e.response.text[:500] + "...")
except json.JSONDecodeError as e:
    print(f"\nError decoding JSON response from token endpoint: {e}")
    if 'token_response' in locals():
       print("Response Text:", token_response.text[:500] + "...")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")