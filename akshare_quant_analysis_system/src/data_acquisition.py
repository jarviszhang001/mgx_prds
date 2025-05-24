import akshare
import pandas as pd
import os

def fetch_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetches historical stock data (OHLCV) for a given stock code and date range.

    Args:
        stock_code (str): The stock code (e.g., "600519" for Kweichow Moutai).
        start_date (str): The start date for the data (format: "YYYYMMDD").
        end_date (str): The end date for the data (format: "YYYYMMDD").

    Returns:
        pd.DataFrame | None: A pandas DataFrame containing the stock data with columns
                             ['date', 'open', 'high', 'low', 'close', 'volume'],
                             or None if an error occurs.
                             The column names from akshare are:
                             日(index), 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
                             These will be mapped to:
                             date, open, close, high, low, volume, turnover, amplitude, change_pct, change_amount, turnover_rate
    """
    try:
        print(f"Fetching stock data for {stock_code} from {start_date} to {end_date}...")
        # stock_zh_a_hist returns: 日期  开盘  收盘  最高  最低  成交量  成交额  振幅  涨跌幅  涨跌额  换手率
        # We need to map these to English names for consistency.
        # We'll also ensure 'date' is a column and not an index.
        stock_data_df = akshare.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # qfq: 前复权 (forward-adjusted)
        )

        if stock_data_df is None or stock_data_df.empty:
            print(f"No data returned for stock code {stock_code}.")
            return None

        # Rename columns to a more standard English format
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "turnover",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "换手率": "turnover_rate"
        }
        stock_data_df.rename(columns=column_mapping, inplace=True)

        # Convert 'date' column to datetime objects
        stock_data_df['date'] = pd.to_datetime(stock_data_df['date'])

        # Select only the required OHLCV columns plus date
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        stock_data_df = stock_data_df[required_columns]

        print(f"Successfully fetched data for {stock_code}.")
        return stock_data_df

    except Exception as e:
        print(f"Error fetching stock data for {stock_code}: {e}")
        return None

def save_data_to_csv(df: pd.DataFrame, file_name: str, directory: str = "../data") -> bool:
    """
    Saves a pandas DataFrame to a CSV file in the specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_name (str): The name of the CSV file (e.g., "stock_data.csv").
        directory (str): The directory to save the file in. Defaults to "../data"
                         (relative to the current script's location in src/).

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    if df is None or df.empty:
        print(f"DataFrame is empty or None. Cannot save to {file_name}.")
        return False

    # Ensure the target directory exists
    # The path is relative to this script's location (src/)
    # So, ../data means one level up from src, then into data.
    # If main.py calls this, os.getcwd() might be different, so careful path handling is needed.
    # For simplicity here, we assume it's run relative to src/ or the path is correctly adjusted.
    
    # Construct the full path relative to this file's location.
    # This makes it more robust if the script is called from different working directories.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir_path = os.path.join(current_script_dir, directory)
    
    if not os.path.exists(target_dir_path):
        try:
            os.makedirs(target_dir_path)
            print(f"Created directory: {target_dir_path}")
        except Exception as e:
            print(f"Error creating directory {target_dir_path}: {e}")
            return False
            
    file_path = os.path.join(target_dir_path, file_name)

    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved successfully to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
        return False

def fetch_index_data(index_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetches historical index data (OHLCV) for a given index code and date range.
    Supported index prefixes include: sh, sz, bj, cy, kc, zg,zs,gz etc.

    Args:
        index_code (str): The index code (e.g., "sh000001" for SSE Composite Index,
                          "sz399001" for SZSE Component Index).
        start_date (str): The start date for the data (format: "YYYYMMDD").
        end_date (str): The end date for the data (format: "YYYYMMDD").

    Returns:
        pd.DataFrame | None: A pandas DataFrame containing the index data,
                             or None if an error occurs.
                             Columns are typically: date, open, high, low, close, volume, turnover.
    """
    try:
        print(f"Fetching index data for {index_code} from {start_date} to {end_date}...")
        # Using index_zh_a_hist as it supports date ranges and provides standard column names.
        # Note: index_code for index_zh_a_hist should be without 'sh' or 'sz' prefix, e.g., "000001" for SSE Composite.
        # However, common usage often includes the prefix, and akshare might handle it.
        # Let's test with common prefixes first (e.g. "sh000001")
        # If that fails, we may need to strip them.
        # The function expects index codes like "000001", "399001", etc.
        
        # Standardizing index code by removing potential "sh" or "sz" prefixes for index_zh_a_hist
        processed_index_code = index_code.replace("sh", "").replace("sz", "")

        index_data_df = akshare.index_zh_a_hist(
            symbol=processed_index_code,
            period="daily",
            start_date=start_date,
            end_date=end_date
        )

        if index_data_df is None or index_data_df.empty:
            print(f"No data returned for index code {index_code} (processed as {processed_index_code}). Akshare might return None or empty DataFrame for invalid codes or network issues.")
            return None

        # Columns from index_zh_a_hist are:
        # 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        # We need to map these to English names for consistency.
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume", # Note: unit is '手' (lots)
            "成交额": "turnover", # Note: unit is '元'
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "换手率": "turnover_rate"
        }
        index_data_df.rename(columns=column_mapping, inplace=True)

        # Convert 'date' column to datetime objects
        index_data_df['date'] = pd.to_datetime(index_data_df['date'])
        
        # Select a subset of columns, similar to fetch_stock_data for basic OHLCV
        # Or decide if all columns from index_zh_a_hist are useful
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        index_data_df = index_data_df[required_columns]

        print(f"Successfully fetched data for index {index_code} (processed as {processed_index_code}).")
        return index_data_df

    except Exception as e:
        print(f"Error fetching index data for {index_code} (processed as {processed_index_code if 'processed_index_code' in locals() else index_code}): {e}")
        return None


if __name__ == '__main__':
    # --- Stock Data Example ---
    ping_an_stock_code = "000001"
    stock_start_date = "20231101"
    stock_end_date = "20231231"
    ping_an_data = fetch_stock_data(stock_code=ping_an_stock_code, start_date=stock_start_date, end_date=stock_end_date)
    if ping_an_data is not None:
        print(f"\n{ping_an_stock_code} Stock Data:")
        print(ping_an_data.head())
        # Save the fetched stock data
        save_success = save_data_to_csv(ping_an_data, f"{ping_an_stock_code}_stock_data_{stock_start_date}_{stock_end_date}.csv")
        if save_success:
            # Verify by trying to read it back (optional, for testing)
            # loaded_df = pd.read_csv(os.path.join("../data", f"{ping_an_stock_code}_stock_data_{stock_start_date}_{stock_end_date}.csv"))
            # print(f"\nLoaded {ping_an_stock_code}_stock_data_{stock_start_date}_{stock_end_date}.csv successfully.")
            # print(loaded_df.head())
            pass


    # Example with a potentially invalid stock code
    invalid_stock_data = fetch_stock_data(stock_code="999999", start_date="20230101", end_date="20230131")
    if invalid_stock_data is None:
        print("\nFailed to fetch data for invalid stock code 999999, as expected.")
    
    # --- Index Data Example ---
    # SSE Composite Index (sh000001 -> 000001 for index_zh_a_hist)
    start_date_index = "20231101"
    end_date_index = "20231231"
    sse_index_code_orig = "sh000001" 
    sse_data = fetch_index_data(index_code=sse_index_code_orig, start_date=start_date_index, end_date=end_date_index)
    if sse_data is not None:
        print(f"\n{sse_index_code_orig} Index Data:")
        print(sse_data.head())
        # Save the fetched index data
        save_data_to_csv(sse_data, f"{sse_index_code_orig}_index_data_{start_date_index}_{end_date_index}.csv")

    # SZSE Component Index (sz399001 -> 399001 for index_zh_a_hist)
    start_date_index_2 = "20240101"
    end_date_index_2 = "20240331"
    szse_index_code_orig = "sz399001"
    szse_data = fetch_index_data(index_code=szse_index_code_orig, start_date=start_date_index_2, end_date=end_date_index_2)
    if szse_data is not None:
        print(f"\n{szse_index_code_orig} Index Data:")
        print(szse_data.head())
        save_data_to_csv(szse_data, f"{szse_index_code_orig}_index_data_{start_date_index_2}_{end_date_index_2}.csv")

    # Example with a potentially invalid index code (for index_zh_a_hist, this might just return empty)
    invalid_index_code_test = "invalid001"
    invalid_index_data = fetch_index_data(index_code=invalid_index_code_test, start_date="20230101", end_date="20230131")
    if invalid_index_data is None:
        print(f"\nFailed to fetch data for invalid index code {invalid_index_code_test}, as expected or it returned empty.")

    # Example of saving an empty dataframe to test save_data_to_csv robustness
    empty_df = pd.DataFrame()
    save_data_to_csv(empty_df, "empty_test_data.csv")
    
    # Example of trying to save None to test save_data_to_csv robustness
    save_data_to_csv(None, "none_test_data.csv")
