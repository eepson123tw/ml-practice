import requests
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Generate list of dates
start_date = datetime.strptime('20211001', '%Y%m%d')
end_date = datetime.strptime('20241001', '%Y%m%d')
date_list = []

current_date = start_date
while current_date <= end_date:
    date_list.append(current_date.strftime('%Y%m%d'))
    current_date += relativedelta(months=1)

# Create the directory if it doesn't exist
os.makedirs('Stock', exist_ok=True)

# Fetch and save data for each date
for date in date_list:
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date}&stockNo=2330"
    response = requests.get(url)
    data = response.json()

    # Check if data is available
    if 'fields' in data and 'data' in data:
        fields = data['fields']
        data_rows = data['data']

        # Create a DataFrame
        df = pd.DataFrame(data_rows, columns=fields)

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join('Stock', f'stock_data_{date}.csv')
        df.to_csv(csv_file_path, index=False)

        print(f"Data for {date} saved to {csv_file_path}")
    else:
        print(f"No data available for {date}")
