import pandas as pd
import numpy as np
import os

def load_data_from_csv(file_path: str) -> pd.DataFrame | None:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame | None: The loaded DataFrame, or None if an error occurs
                             (e.g., file not found).
    """
    try:
        # The file_path is expected to be relative to the project root,
        # or an absolute path.
        # main.py (at project root) will call this with paths like "data/somefile.csv".
        # If this script is run directly from src/, paths like "../data/somefile.csv" would be used.

        # Check if the provided file_path exists as is (common when main.py is CWD)
        if not os.path.exists(file_path):
            # If not, try to construct path relative to this script's project root
            # This helps if a function within src/ calls load_data_from_csv with a simple name
            # expecting it to be in the project's data/ folder.
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_script_dir)
            potential_path = os.path.join(project_root, file_path)
            
            if os.path.exists(potential_path):
                file_path = potential_path
            else:
                # If still not found, raise FileNotFoundError (or let pd.read_csv raise it)
                # For clarity, we'll let pd.read_csv handle the final error if path is truly bad.
                pass # Fall through to pd.read_csv which will error if path is invalid

        print(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Ensure 'date' column is parsed as datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        print(f"Successfully loaded data from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File at {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def handle_missing_values(df: pd.DataFrame, strategy: str = 'ffill') -> pd.DataFrame:
    """
    Handles missing values (NaN) in a DataFrame using a specified strategy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): The strategy for handling missing values.
                        Options: 'ffill' (forward fill), 'bfill' (backward fill),
                                 'dropna' (drop rows with any NaNs),
                                 'mean' (fill with column mean - for numeric columns only).
                        Defaults to 'ffill'.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    if df is None:
        print("Input DataFrame is None. Cannot handle missing values.")
        return None # Or raise error, or return an empty DataFrame
    
    processed_df = df.copy()
    if strategy == 'ffill':
        processed_df.ffill(inplace=True)
        print("Missing values handled using forward fill (ffill).")
    elif strategy == 'bfill':
        processed_df.bfill(inplace=True)
        print("Missing values handled using backward fill (bfill).")
    elif strategy == 'dropna':
        processed_df.dropna(inplace=True)
        print("Rows with missing values dropped (dropna).")
    elif strategy == 'mean':
        # Iterate over columns and apply mean imputation only to numeric columns
        for col in processed_df.columns:
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
        print("Missing values in numeric columns handled using mean imputation.")
    else:
        print(f"Warning: Unknown missing value strategy '{strategy}'. DataFrame returned unchanged.")
    return processed_df

def calculate_daily_returns(df: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
    """
    Calculates daily percentage returns from a specified price column.

    Args:
        df (pd.DataFrame): DataFrame with price data. Must contain the price_column.
                           The DataFrame should be sorted by date in ascending order.
        price_column (str): The name of the column containing price data.
                            Defaults to 'close'.

    Returns:
        pd.DataFrame: The DataFrame with a new 'daily_return' column.
                      Returns the original DataFrame if price_column is not found
                      or if an error occurs.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot calculate daily returns.")
        return df
    
    if price_column not in df.columns:
        print(f"Error: Price column '{price_column}' not found in DataFrame.")
        return df

    processed_df = df.copy()
    # Ensure price column is numeric and handle potential non-numeric values if necessary
    if not pd.api.types.is_numeric_dtype(processed_df[price_column]):
        print(f"Warning: Price column '{price_column}' is not numeric. Attempting conversion.")
        processed_df[price_column] = pd.to_numeric(processed_df[price_column], errors='coerce')
        # NaNs introduced here will correctly propagate in the calculation below

    # Calculate daily returns explicitly to control NaN propagation
    # (price_t / price_{t-1}) - 1  OR  (price_t - price_{t-1}) / price_{t-1}
    # Using diff and shift for clarity on NaN propagation
    diffs = processed_df[price_column].diff()
    shifted_prices = processed_df[price_column].shift(1)
    
    # Calculate returns. This will be NaN where shifted_prices is NaN (i.e., first row)
    # or where shifted_prices is 0 and diffs is non-zero (inf)
    # or where diffs is NaN (e.g. if price_column had consecutive NaNs after conversion)
    returns = diffs / shifted_prices
    
    processed_df['daily_return'] = returns

    # Handle potential inf/-inf values that arise from division by zero.
    processed_df['daily_return'] = processed_df['daily_return'].replace([np.inf, -np.inf], np.nan)
    
    print(f"Daily returns calculated based on '{price_column}' column and added as 'daily_return'.")
    return processed_df

def calculate_moving_average(df: pd.DataFrame, window: int, price_column: str = 'close') -> pd.DataFrame:
    """
    Calculates the moving average for a specified price column.

    Args:
        df (pd.DataFrame): DataFrame with price data. Must contain the price_column.
                           The DataFrame should be sorted by date in ascending order.
        window (int): The window size for the moving average (e.g., 20 for 20-day MA).
        price_column (str): The name of the column containing price data.
                            Defaults to 'close'.

    Returns:
        pd.DataFrame: The DataFrame with a new moving average column (e.g., 'MA_20').
                      Returns the original DataFrame if price_column is not found,
                      window is non-positive, or an error occurs.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot calculate moving average.")
        return df

    if price_column not in df.columns:
        print(f"Error: Price column '{price_column}' not found in DataFrame.")
        return df
    
    if not isinstance(window, int) or window <= 0:
        print(f"Error: Window size must be a positive integer. Received: {window}")
        return df

    processed_df = df.copy()
    # Ensure price column is numeric
    if not pd.api.types.is_numeric_dtype(processed_df[price_column]):
        print(f"Warning: Price column '{price_column}' is not numeric. Attempting conversion.")
        processed_df[price_column] = pd.to_numeric(processed_df[price_column], errors='coerce')
        # NaNs from conversion will propagate in .rolling().mean()

    ma_column_name = f'MA_{window}'
    processed_df[ma_column_name] = processed_df[price_column].rolling(window=window, min_periods=1).mean()
    # min_periods=1 ensures that we get a value even if there are fewer than `window` periods,
    # which is common at the beginning of the series.
    
    print(f"Moving average with window {window} calculated for '{price_column}' and added as '{ma_column_name}'.")
    return processed_df

def calculate_rsi(df: pd.DataFrame, window: int = 14, price_column: str = 'close') -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): DataFrame with price data. Must contain the price_column.
                           The DataFrame should be sorted by date in ascending order.
        window (int): The window size for RSI calculation (typically 14).
        price_column (str): The name of the column containing price data.
                            Defaults to 'close'.

    Returns:
        pd.DataFrame: The DataFrame with a new 'RSI_{window}' column.
                      Returns the original DataFrame if price_column is not found,
                      window is non-positive, or an error occurs.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot calculate RSI.")
        return df

    if price_column not in df.columns:
        print(f"Error: Price column '{price_column}' not found in DataFrame.")
        return df

    if not isinstance(window, int) or window <= 0:
        print(f"Error: Window size must be a positive integer. Received: {window}")
        return df

    processed_df = df.copy()
    # Ensure price column is numeric
    if not pd.api.types.is_numeric_dtype(processed_df[price_column]):
        print(f"Warning: Price column '{price_column}' is not numeric. Attempting conversion.")
        processed_df[price_column] = pd.to_numeric(processed_df[price_column], errors='coerce')
        # Handle NaNs that might have been introduced by coercion or were already present
        if processed_df[price_column].isnull().any():
            print(f"Warning: NaNs present in price column '{price_column}' after potential conversion. RSI might be affected.")
            # Optionally, fill NaNs here using a strategy if desired, e.g., ffill
            # processed_df[price_column] = processed_df[price_column].ffill()


    delta = processed_df[price_column].diff(1)
    delta.dropna(inplace=True) # Remove first NaN from diff

    gain = delta.copy()
    loss = delta.copy()

    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss) # Make losses positive for calculation

    # Calculate average gain and loss using Wilder's smoothing method (exponential moving average)
    # For the first value, simple moving average is used.
    avg_gain = gain.rolling(window=window, min_periods=window).mean().fillna(0)
    avg_loss = loss.rolling(window=window, min_periods=window).mean().fillna(0)
    
    # For subsequent values, use Wilder's smoothing:
    # AvgGain_current = ((AvgGain_previous * (window - 1)) + Gain_current) / window
    # This is equivalent to an EMA with alpha = 1/window
    # Pandas ewm with com = window - 1 (alpha=1/(1+com)) is similar to Wilder's.
    # For direct Wilder's as often described:
    if len(avg_gain) > window: # Check if there are enough data points beyond the initial SMA
        for i in range(window, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window - 1) + gain.iloc[i]) / window
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window - 1) + loss.iloc[i]) / window
    
    # Alternative using pandas ewm, which is more common in modern libraries
    # avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    # avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()


    rs = avg_gain / avg_loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True) # handle division by zero if avg_loss is 0
    rs.fillna(0, inplace=True) # Fill NaN RS with 0 (implies no loss, so RSI would be 100, or no gain, RSI 0) - this needs care.
                               # A common approach is to fillna for RS when avg_loss is 0, leading to RSI 100.

    rsi = 100 - (100 / (1 + rs))
    
    # If avg_loss is 0, RS is inf, RSI is 100. If avg_gain is also 0, RS is NaN (0/0), RSI is often taken as 50 or handled.
    # Correcting cases where avg_loss can be 0
    rsi[avg_loss == 0] = 100 # If avg_loss is 0, RSI is 100 (unless avg_gain is also 0)
    rsi[ (avg_gain == 0) & (avg_loss == 0) ] = 0 # Or 50, depends on convention. StockCharts uses 0.

    rsi_column_name = f'RSI_{window}'
    processed_df[rsi_column_name] = rsi
    
    # The RSI will have NaNs for the first `window` periods due to rolling mean and diff.
    print(f"RSI with window {window} calculated for '{price_column}' and added as '{rsi_column_name}'.")
    return processed_df

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: int = 2, price_column: str = 'close') -> pd.DataFrame:
    """
    Calculates Bollinger Bands.

    Args:
        df (pd.DataFrame): DataFrame with price data. Must contain the price_column.
                           The DataFrame should be sorted by date in ascending order.
        window (int): The window size for the moving average and standard deviation (typically 20).
        num_std_dev (int): The number of standard deviations for the bands (typically 2).
        price_column (str): The name of the column to use for calculation. Defaults to 'close'.

    Returns:
        pd.DataFrame: The DataFrame with new columns: 'BB_Mid_{window}', 
                      'BB_Upper_{window}', 'BB_Lower_{window}'.
                      Returns the original DataFrame if errors occur.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot calculate Bollinger Bands.")
        return df

    if price_column not in df.columns:
        print(f"Error: Price column '{price_column}' not found in DataFrame.")
        return df

    if not isinstance(window, int) or window <= 0:
        print(f"Error: Window size must be a positive integer. Received: {window}")
        return df
        
    if not isinstance(num_std_dev, (int, float)) or num_std_dev <= 0:
        print(f"Error: Number of standard deviations must be a positive number. Received: {num_std_dev}")
        return df

    processed_df = df.copy()
    # Ensure price column is numeric
    if not pd.api.types.is_numeric_dtype(processed_df[price_column]):
        print(f"Warning: Price column '{price_column}' is not numeric. Attempting conversion.")
        processed_df[price_column] = pd.to_numeric(processed_df[price_column], errors='coerce')
        # NaNs from conversion will propagate.

    middle_band_name = f'BB_Mid_{window}'
    upper_band_name = f'BB_Upper_{window}'
    lower_band_name = f'BB_Lower_{window}'

    # Calculate the middle band (simple moving average)
    processed_df[middle_band_name] = processed_df[price_column].rolling(window=window, min_periods=1).mean()
    
    # Calculate the standard deviation
    std_dev = processed_df[price_column].rolling(window=window, min_periods=1).std()
    
    # Calculate upper and lower bands
    processed_df[upper_band_name] = processed_df[middle_band_name] + (std_dev * num_std_dev)
    processed_df[lower_band_name] = processed_df[middle_band_name] - (std_dev * num_std_dev)
    
    print(f"Bollinger Bands (window {window}, std_dev {num_std_dev}) calculated for '{price_column}'.")
    return processed_df


if __name__ == '__main__':
    # This block is for example usage and basic testing.
    # It assumes that you might have some sample data in ../data/
    # For robust testing, dedicated test files (e.g., in tests/) are better.

    print("--- Data Processing Module Example ---")

    # Create a sample DataFrame for demonstration (longer for better RSI/BB results)
    date_rng = pd.date_range(start='2023-01-01', end='2023-02-01', freq='B') # Business days
    sample_size = len(date_rng)
    np.random.seed(42) # for reproducibility
    close_prices = 100 + np.random.randn(sample_size).cumsum()
    
    sample_data_full = {
        'date': date_rng,
        'open': close_prices - np.random.rand(sample_size) * 0.5,
        'high': close_prices + np.random.rand(sample_size) * 0.5,
        'low': close_prices - np.random.rand(sample_size) * 0.6, # Ensure low is usually lower than open
        'close': close_prices,
        'volume': np.random.randint(1000, 5000, size=sample_size)
    }
    sample_df_full = pd.DataFrame(sample_data_full)
    # Ensure low is not higher than high or close, and high not lower than close
    sample_df_full['low'] = sample_df_full[['low', 'open', 'close', 'high']].min(axis=1)
    sample_df_full['high'] = sample_df_full[['high', 'open', 'close']].max(axis=1)


    print("\nOriginal Full Sample DataFrame (first 5 rows):")
    print(sample_df_full.head())
    
    # Add some NaNs for testing handle_missing_values
    sample_df_with_nans = sample_df_full.copy()
    sample_df_with_nans.loc[sample_df_with_nans.index[2], 'close'] = np.nan
    sample_df_with_nans.loc[sample_df_with_nans.index[5], 'volume'] = np.nan
    sample_df_with_nans.loc[sample_df_with_nans.index[7], 'open'] = np.nan
    print("\nOriginal Sample DataFrame with NaNs (selected rows):")
    print(sample_df_with_nans[sample_df_with_nans.isnull().any(axis=1)].head())


    # Test handle_missing_values
    df_ffill = handle_missing_values(sample_df_with_nans.copy(), strategy='ffill')
    print("\nDataFrame after ffill (rows that had NaNs):")
    # Show how NaNs were filled for the specific rows modified
    print(df_ffill.loc[sample_df_with_nans.index[[2, 5, 7]]])


    # Test calculate_daily_returns (using the ffill_df as it has no NaNs in 'close')
    df_returns = calculate_daily_returns(df_ffill.copy(), price_column='close')
    print("\nDataFrame with Daily Returns (first 5 rows):")
    print(df_returns[['date', 'close', 'daily_return']].head())

    # Test calculate_moving_average
    df_ma = calculate_moving_average(df_returns.copy(), window=5, price_column='close')
    print("\nDataFrame with 5-day Moving Average (first 5 rows):")
    print(df_ma[['date', 'close', 'MA_5', 'daily_return']].head())

    # Test calculate_rsi
    df_rsi = calculate_rsi(df_ma.copy(), window=14, price_column='close')
    print("\nDataFrame with RSI_14 (rows around where RSI becomes valid):")
    print(df_rsi[['date', 'close', 'RSI_14']].iloc[10:20]) # Show some initial NaNs and then values

    # Test calculate_bollinger_bands
    df_bb = calculate_bollinger_bands(df_rsi.copy(), window=20, num_std_dev=2, price_column='close')
    print("\nDataFrame with Bollinger Bands (window 20, std 2) (rows around where BB becomes valid):")
    print(df_bb[['date', 'close', 'BB_Mid_20', 'BB_Upper_20', 'BB_Lower_20']].iloc[15:25])
    
    # --- File Loading Example ---
    temp_data_dir_name = "temp_data_for_processing_example"
    current_script_dir_for_save = os.path.dirname(os.path.abspath(__file__))
    project_root_for_save = os.path.dirname(current_script_dir_for_save)
    temp_data_full_dir = os.path.join(project_root_for_save, temp_data_dir_name)
    
    if not os.path.exists(temp_data_full_dir):
        os.makedirs(temp_data_full_dir)
    
    sample_csv_path = os.path.join(temp_data_full_dir, "sample_stock_data_full.csv")
    sample_df_full.to_csv(sample_csv_path, index=False)
    print(f"\nSaved full sample data to {sample_csv_path} for load_data_from_csv example.")

    loaded_df = load_data_from_csv(f"{temp_data_dir_name}/sample_stock_data_full.csv") # Path relative to project root
    if loaded_df is not None:
        print("\nSuccessfully loaded DataFrame for file testing (first 5 rows):")
        print(loaded_df.head())
        
        loaded_df_processed = calculate_moving_average(loaded_df, window=5, price_column='close')
        loaded_df_processed = calculate_daily_returns(loaded_df_processed, price_column='close')
        loaded_df_processed = calculate_rsi(loaded_df_processed, window=14)
        loaded_df_processed = calculate_bollinger_bands(loaded_df_processed, window=20)
        print("\nProcessed loaded DataFrame with all indicators (first few rows where indicators are valid):")
        print(loaded_df_processed.dropna().head()) # Drop NA rows for cleaner display of calculated values
    else:
        print("\nFailed to load sample_stock_data_full.csv for testing.")

    # Test with a non-existent file
    print("\n--- Test loading non-existent file ---")
    non_existent_df = load_data_from_csv("non_existent_file_123.csv")
    if non_existent_df is None:
        print("Correctly handled non-existent file: Returned None.")

    # Clean up the temporary CSV file and directory
    try:
        if os.path.exists(sample_csv_path):
            os.remove(sample_csv_path)
        if os.path.exists(temp_data_full_dir):
            # Only remove if empty, or use shutil.rmtree if it might contain other things (be careful)
            if not os.listdir(temp_data_full_dir): 
                os.rmdir(temp_data_full_dir)
            else:
                print(f"Warning: Directory {temp_data_full_dir} not empty, not removing.")
        print(f"\nCleaned up or attempted cleanup of temporary file/directory: {temp_data_full_dir}")
    except OSError as e:
        print(f"Error during cleanup: {e}")

    print("\n--- End of Data Processing Module Example ---")
