import pandas as pd
import numpy as np
# Assuming data_processing.py is in the same directory or PYTHONPATH is set up
# For direct relative import if files are in the same package:
# from . import data_processing
# For now, let's assume we might need to load data and process it here for examples,
# so direct imports from data_processing will be used in the __main__ block.

def generate_sma_signals(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Generates trading signals based on Simple Moving Average (SMA) crossover.

    Args:
        df (pd.DataFrame): DataFrame assumed to have 'close' prices.
                           It's expected that MA columns (e.g., 'MA_5', 'MA_20')
                           will be calculated if not already present.
                           This function will look for f'MA_{short_window}' and f'MA_{long_window}'.
        short_window (int): The window size for the short-term SMA.
        long_window (int): The window size for the long-term SMA.

    Returns:
        pd.DataFrame: The DataFrame with a new 'sma_signal' column.
                      (1 for Buy, -1 for Sell, 0 for Hold).
                      Returns the original DataFrame if required MA columns are missing.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot generate SMA signals.")
        return df

    short_ma_col = f'MA_{short_window}'
    long_ma_col = f'MA_{long_window}'

    if short_ma_col not in df.columns or long_ma_col not in df.columns:
        print(f"Error: Required MA columns ('{short_ma_col}', '{long_ma_col}') not found.")
        print("Please ensure moving averages are calculated and present in the DataFrame.")
        print("You can use `calculate_moving_average` from the data_processing module.")
        return df

    processed_df = df.copy()
    processed_df['sma_signal'] = 0 # Default to Hold signal

    # Buy signal: Short MA crosses above Long MA
    # (Short MA > Long MA) & (Short MA.shift(1) < Long MA.shift(1))
    buy_condition = (processed_df[short_ma_col] > processed_df[long_ma_col]) & \
                    (processed_df[short_ma_col].shift(1) < processed_df[long_ma_col].shift(1))
    processed_df.loc[buy_condition, 'sma_signal'] = 1

    # Sell signal: Short MA crosses below Long MA
    # (Short MA < Long MA) & (Short MA.shift(1) > Long MA.shift(1))
    sell_condition = (processed_df[short_ma_col] < processed_df[long_ma_col]) & \
                     (processed_df[short_ma_col].shift(1) > processed_df[long_ma_col].shift(1))
    processed_df.loc[sell_condition, 'sma_signal'] = -1
    
    # Forward fill signals for positions held until next signal
    # This is a common way to represent holding a position after a signal.
    # However, for discrete crossover events, the above is sufficient.
    # If we want to represent a "state" (e.g. in bullish mode or bearish mode):
    # processed_df['sma_signal'] = processed_df['sma_signal'].replace(to_replace=0, method='ffill').fillna(0)

    print(f"SMA signals generated based on {short_ma_col} and {long_ma_col} crossover.")
    return processed_df

def generate_rsi_signals(df: pd.DataFrame, rsi_window: int, 
                         oversold_threshold: int = 30, 
                         overbought_threshold: int = 70) -> pd.DataFrame:
    """
    Generates trading signals based on Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): DataFrame assumed to have an RSI column 
                           (e.g., 'RSI_14'). This function will look for f'RSI_{rsi_window}'.
        rsi_window (int): The window size used for the RSI calculation (e.g., 14).
        oversold_threshold (int): The RSI level below which the asset is considered oversold.
        overbought_threshold (int): The RSI level above which the asset is considered overbought.

    Returns:
        pd.DataFrame: The DataFrame with a new 'rsi_signal' column.
                      (1 for Buy, -1 for Sell, 0 for Hold).
                      Returns the original DataFrame if the required RSI column is missing.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot generate RSI signals.")
        return df

    rsi_col = f'RSI_{rsi_window}'

    if rsi_col not in df.columns:
        print(f"Error: Required RSI column '{rsi_col}' not found.")
        print("Please ensure RSI is calculated and present in the DataFrame.")
        print("You can use `calculate_rsi` from the data_processing module.")
        return df

    processed_df = df.copy()
    processed_df['rsi_signal'] = 0 # Default to Hold signal

    # Buy signal: RSI crosses above oversold threshold from below
    buy_condition = (processed_df[rsi_col] > oversold_threshold) & \
                    (processed_df[rsi_col].shift(1) <= oversold_threshold)
    processed_df.loc[buy_condition, 'rsi_signal'] = 1

    # Sell signal: RSI crosses below overbought threshold from above
    sell_condition = (processed_df[rsi_col] < overbought_threshold) & \
                     (processed_df[rsi_col].shift(1) >= overbought_threshold)
    processed_df.loc[sell_condition, 'rsi_signal'] = -1

    print(f"RSI signals generated based on {rsi_col} with thresholds {oversold_threshold}/{overbought_threshold}.")
    return processed_df

def generate_bollinger_band_signals(df: pd.DataFrame, bb_window: int, price_column: str = 'close') -> pd.DataFrame:
    """
    Generates trading signals based on Bollinger Bands.

    Args:
        df (pd.DataFrame): DataFrame with price data and Bollinger Band columns.
                           Expected columns: price_column (e.g. 'close'),
                           f'BB_Lower_{bb_window}', f'BB_Upper_{bb_window}'.
        bb_window (int): The window size used for Bollinger Bands calculation.
        price_column (str): The name of the price column. Defaults to 'close'.

    Returns:
        pd.DataFrame: The DataFrame with a new 'bb_signal' column.
                      (1 for Buy, -1 for Sell, 0 for Hold).
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot generate Bollinger Band signals.")
        return df

    lower_band_col = f'BB_Lower_{bb_window}'
    upper_band_col = f'BB_Upper_{bb_window}'

    if price_column not in df.columns or lower_band_col not in df.columns or upper_band_col not in df.columns:
        print(f"Error: Required columns ('{price_column}', '{lower_band_col}', '{upper_band_col}') not found.")
        print("Please ensure Bollinger Bands are calculated and present in the DataFrame.")
        print("You can use `calculate_bollinger_bands` from the data_processing module.")
        return df
        
    processed_df = df.copy()
    processed_df['bb_signal'] = 0 # Default to Hold

    # Buy signal: Price crosses below lower band, then crosses back above lower band.
    # Condition 1: Price was below or equal to lower band in previous period
    # Condition 2: Price is above lower band in current period
    buy_crossed_back_above_lower = (processed_df[price_column].shift(1) <= processed_df[lower_band_col].shift(1)) & \
                                   (processed_df[price_column] > processed_df[lower_band_col])
    processed_df.loc[buy_crossed_back_above_lower, 'bb_signal'] = 1

    # Sell signal: Price crosses above upper band, then crosses back below upper band.
    # Condition 1: Price was above or equal to upper band in previous period
    # Condition 2: Price is below upper band in current period
    sell_crossed_back_below_upper = (processed_df[price_column].shift(1) >= processed_df[upper_band_col].shift(1)) & \
                                    (processed_df[price_column] < processed_df[upper_band_col])
    processed_df.loc[sell_crossed_back_below_upper, 'bb_signal'] = -1
    
    print(f"Bollinger Band signals generated for {price_column} using bands for window {bb_window}.")
    return processed_df

def combine_signals(df: pd.DataFrame, signal_columns: list[str], strategy: str = 'majority') -> pd.DataFrame:
    """
    Combines multiple signal columns into a final trading signal.

    Args:
        df (pd.DataFrame): DataFrame containing individual signal columns.
        signal_columns (list[str]): A list of column names that contain signals (1, -1, or 0).
        strategy (str): The strategy for combining signals.
                        'majority': Takes the most frequent signal (1, -1, or 0).
                                    If 1 and -1 have equal frequency, it results in a hold (0).
                                    If 0 is the most frequent or tied for most frequent with a directional signal,
                                    it results in 0 unless the other signal has clear majority.
                        'unanimous': Requires all signals to agree (all 1 for Buy, all -1 for Sell).
                                     Otherwise, Hold (0).
    Returns:
        pd.DataFrame: The DataFrame with a new 'combined_signal' column.
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot combine signals.")
        return df

    if not signal_columns:
        print("Error: No signal columns provided for combination.")
        df['combined_signal'] = 0
        return df

    missing_cols = [col for col in signal_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing signal columns: {missing_cols}. Cannot combine signals.")
        return df

    processed_df = df.copy()
    
    if strategy == 'majority':
        def majority_vote(row):
            votes = row[signal_columns].value_counts()
            # Check if 0 (hold) is the most frequent or tied for most frequent with a directional signal
            if 0 in votes and votes[0] == votes.max():
                 # If 0 is strictly more frequent than any other signal, it's a hold.
                 # Or if 0 is tied with 1 and -1 is less, or 0 tied with -1 and 1 is less.
                 # Essentially, if 0 is the mode, or if 1 and -1 don't form a clear majority over 0.
                 if votes.get(1,0) == votes.get(-1,0): # e.g. [1, -1, 0] -> 0
                     return 0
                 if votes.get(1,0) > votes.get(-1,0) and votes.get(1,0) > votes.get(0,0):
                     return 1
                 if votes.get(-1,0) > votes.get(1,0) and votes.get(-1,0) > votes.get(0,0):
                     return -1
                 return 0 # Default to hold in ambiguous cases or if 0 is strongest
            
            # If directional signals (1 or -1) are present and are the mode
            if votes.get(1, 0) > votes.get(-1, 0) and votes.get(1, 0) >= votes.get(0, 0):
                return 1
            elif votes.get(-1, 0) > votes.get(1, 0) and votes.get(-1, 0) >= votes.get(0, 0):
                return -1
            else: # Handles ties between 1 and -1, or if 0 is the majority
                return 0

        processed_df['combined_signal'] = processed_df.apply(majority_vote, axis=1)
        print("Signals combined using 'majority' vote strategy.")

    elif strategy == 'unanimous':
        def unanimous_vote(row):
            signals = row[signal_columns]
            if (signals == 1).all():
                return 1
            elif (signals == -1).all():
                return -1
            else:
                return 0
        processed_df['combined_signal'] = processed_df.apply(unanimous_vote, axis=1)
        print("Signals combined using 'unanimous' agreement strategy.")
        
    else:
        print(f"Warning: Unknown combination strategy '{strategy}'. 'combined_signal' column will be all 0s.")
        processed_df['combined_signal'] = 0
        
    return processed_df


if __name__ == '__main__':
    # This block is for example usage and basic testing.
    # It requires data_processing.py to be accessible for imports.
    
    # Ensure src directory (parent of current_dir) is in sys.path for module import
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__)) # analysis_engine.py is in src/
    project_root = os.path.dirname(current_dir) # This should be akshare_quant_analysis_system/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from src import data_processing # Now 'from src import data_processing' should work

    print("--- Analysis Engine Module Example ---")

    # Create a sample DataFrame (similar to data_processing.py for consistency)
    date_rng = pd.date_range(start='2023-01-01', end='2023-04-01', freq='B') # Extended for more signals
    sample_size = len(date_rng)
    np.random.seed(42)
    close_prices = 100 + np.random.randn(sample_size).cumsum()
    
    sample_data_full = {
        'date': date_rng,
        'close': close_prices,
        'open': close_prices - np.random.rand(sample_size) * 0.5,
        'high': close_prices + np.random.rand(sample_size) * 0.5,
        'low': close_prices - np.random.rand(sample_size) * 0.6,
        'volume': np.random.randint(1000, 5000, size=sample_size)
    }
    sample_df = pd.DataFrame(sample_data_full)
    sample_df['low'] = sample_df[['low', 'open', 'close', 'high']].min(axis=1)
    sample_df['high'] = sample_df[['high', 'open', 'close']].max(axis=1)
    
    print("\nOriginal Sample DataFrame (first 5 rows):")
    print(sample_df.head())

    # 1. Process data using functions from data_processing
    df_processed = sample_df.copy()
    
    # SMA
    short_sma_window = 5
    long_sma_window = 20
    df_processed = data_processing.calculate_moving_average(df_processed, window=short_sma_window, price_column='close')
    df_processed = data_processing.calculate_moving_average(df_processed, window=long_sma_window, price_column='close')
    
    # RSI
    rsi_calc_window = 14
    df_processed = data_processing.calculate_rsi(df_processed, window=rsi_calc_window, price_column='close')

    # Bollinger Bands
    bb_calc_window = 20
    df_processed = data_processing.calculate_bollinger_bands(df_processed, window=bb_calc_window, price_column='close')

    print(f"\nDataFrame after calculating indicators (Tail to see values):")
    print(df_processed.tail())

    # 2. Generate signals
    df_with_signals = df_processed.copy()
    df_with_signals = generate_sma_signals(df_with_signals, short_window=short_sma_window, long_window=long_sma_window)
    df_with_signals = generate_rsi_signals(df_with_signals, rsi_window=rsi_calc_window, oversold_threshold=30, overbought_threshold=70)
    df_with_signals = generate_bollinger_band_signals(df_with_signals, bb_window=bb_calc_window, price_column='close')
    
    print(f"\nDataFrame with Individual Signals (SMA, RSI, BB) (showing rows with any signal):")
    signal_cols_for_display = ['date', 'close', 'sma_signal', 'rsi_signal', 'bb_signal']
    print(df_with_signals[signal_cols_for_display][(df_with_signals['sma_signal'] != 0) | (df_with_signals['rsi_signal'] != 0) | (df_with_signals['bb_signal'] != 0)].head(10))

    # 3. Combine signals
    signal_columns_to_combine = ['sma_signal', 'rsi_signal', 'bb_signal']
    
    # Majority strategy
    df_combined_majority = combine_signals(df_with_signals.copy(), signal_columns=signal_columns_to_combine, strategy='majority')
    print(f"\nDataFrame with Combined Signals ('majority') (showing rows with any original or combined signal):")
    combined_cols_for_display_maj = signal_cols_for_display + ['combined_signal']
    print(df_combined_majority[combined_cols_for_display_maj][(df_combined_majority['sma_signal'] != 0) | (df_combined_majority['rsi_signal'] != 0) | (df_combined_majority['bb_signal'] != 0) | (df_combined_majority['combined_signal'] !=0)].head(15))

    # Unanimous strategy
    df_combined_unanimous = combine_signals(df_with_signals.copy(), signal_columns=signal_columns_to_combine, strategy='unanimous')
    print(f"\nDataFrame with Combined Signals ('unanimous') (showing rows with any original or combined signal):")
    combined_cols_for_display_uni = signal_cols_for_display + ['combined_signal']
    print(df_combined_unanimous[combined_cols_for_display_uni][(df_combined_unanimous['sma_signal'] != 0) | (df_combined_unanimous['rsi_signal'] != 0) | (df_combined_unanimous['bb_signal'] != 0) | (df_combined_unanimous['combined_signal'] !=0)].head(15))
    
    print("\n--- End of Analysis Engine Module Example ---")
