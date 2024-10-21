import requests
import pandas as pd
import plotly.express as px

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 確認請求成功
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def process_data(data):
    # 將 JSON 資料轉換為 DataFrame
    df = pd.DataFrame(data)
    
    # 將數值欄位轉換為數字類型，移除逗號並處理正負號
    numeric_columns = [
        'TodayLimitUp', 'TodayOpeningRefPrice', 'TodayLimitDown',
        'PreviousDayOpeningRefPrice', 'PreviousDayPrice',
        'PreviousDayLimitUp', 'PreviousDayLimitDown'
    ]
    
    for col in numeric_columns:
        # 移除逗號、'X'、'+' 和 '-'，並處理漲跌價差中的負號
        df[col] = df[col].str.replace(',', '').replace('X', '').replace('+', '').replace('-', '')
        df[col] = df[col].apply(lambda x: f"-{x}" if '-' in x else x)
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 將無法轉換的值設為 NaN
    
    # 處理 'LastTradingDay' 欄位，將民國年轉換為西元年
    if 'LastTradingDay' in df.columns:
        df['LastTradingDay'] = df['LastTradingDay'].apply(convert_roc_date)
    
    return df

def convert_roc_date(roc_date_str):
    """
    將民國年日期轉換為西元年日期格式。
    例如：'1131018' 轉換為 '2024/10/18'
    假設格式為 'YYYMMDD'，需要根據實際情況調整
    """
    try:
        roc_year = int(roc_date_str[:3])
        month = roc_date_str[3:5]
        day = roc_date_str[5:7]
        year = roc_year + 1911
        return f"{year}/{month}/{day}"
    except:
        return roc_date_str  # 若格式不正確，返回原始字串

def visualize_data_plotly(df):
    # 範例 1: 比較各基金的今日漲停和跌停（多線圖）
    fig1 = px.line(df, x='Name', y=['TodayLimitUp', 'TodayLimitDown'],
                   title='今日漲停與跌停比較',
                   labels={'value': '價格', 'variable': '類型'},
                   markers=True)
    fig1.update_layout(xaxis_title='基金名稱', yaxis_title='價格')
    fig1.show()
    
    # 範例 2: 比較各基金的今日開盤參考價與昨日收盤價（多線圖）
    df_melt = df.melt(id_vars=['Name'], value_vars=['TodayOpeningRefPrice', 'PreviousDayPrice'],
                      var_name='價格類型', value_name='價格')
    fig2 = px.line(df_melt, x='Name', y='價格', color='價格類型',
                   title='今日開盤參考價 vs 昨日收盤價',
                   markers=True)
    fig2.update_layout(xaxis_title='基金名稱', yaxis_title='價格')
    fig2.show()

    # 範例 3: 多線圖展示多個價格類型
    df_long = df.melt(id_vars=['Name'], value_vars=['TodayLimitUp', 'TodayLimitDown',
                                                   'TodayOpeningRefPrice', 'PreviousDayPrice'],
                      var_name='價格類型', value_name='價格')
    fig3 = px.line(df_long, x='Name', y='價格', color='價格類型',
                   title='各種價格類型比較',
                   markers=True)
    fig3.update_layout(xaxis_title='基金名稱', yaxis_title='價格')
    fig3.show()

def main():
    # API URL（請根據實際情況修改）
    url = 'https://openapi.twse.com.tw/v1/exchangeReport/TWT84U'
    
    # 獲取資料
    data = fetch_data(url)
    
    if data:
        # 處理資料
        df = process_data(data)
        
        # 限制為前 10 筆資料
        df = df.head(10)
        
        # 顯示前幾行資料
        print(df.head())
        
        # 可視化資料
        visualize_data_plotly(df)
    else:
        print("無法取得資料。")

if __name__ == "__main__":
    main()
