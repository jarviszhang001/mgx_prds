import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import os
import sys

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src import data_processing

@pytest.fixture
def sample_df_with_nans():
    """DataFrame with NaNs for testing missing value handling."""
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'A': [1.0, np.nan, 3.0, np.nan, 5.0],
        'B': [np.nan, 2.0, np.nan, 4.0, np.nan],
        'C': [10.0, 20.0, 30.0, 40.0, 50.0] 
    }
    return pd.DataFrame(data).set_index('date')

@pytest.fixture
def sample_price_df():
    """DataFrame for testing price-based calculations like returns, MA, RSI, BB."""
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                                '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20']),
        'close': [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0, 18.0, 19.0, 17.0, 16.0, 18.0, 20.0, 22.0, 21.0, 23.0]
    }
    df = pd.DataFrame(data)
    df['close'] = df['close'].astype(float) 
    return df.set_index('date')


@pytest.fixture
def rsi_fixture_sample_df(): 
    prices = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64  
    ]
    dates = pd.date_range(start='2023-01-01', periods=len(prices), freq='B')
    return pd.DataFrame({'date': dates, 'close': prices}).set_index('date')


# Tests for handle_missing_values
def test_handle_missing_values_ffill(sample_df_with_nans):
    df_ffill = data_processing.handle_missing_values(sample_df_with_nans.copy(), strategy='ffill')
    expected_A = pd.Series([1.0, 1.0, 3.0, 3.0, 5.0], index=sample_df_with_nans.index, name='A', dtype=float)
    expected_B = pd.Series([np.nan, 2.0, 2.0, 4.0, 4.0], index=sample_df_with_nans.index, name='B', dtype=float)
    assert_series_equal(df_ffill['A'], expected_A)
    assert_series_equal(df_ffill['B'], expected_B)

def test_handle_missing_values_bfill(sample_df_with_nans):
    df_bfill = data_processing.handle_missing_values(sample_df_with_nans.copy(), strategy='bfill')
    expected_A = pd.Series([1.0, 3.0, 3.0, 5.0, 5.0], index=sample_df_with_nans.index, name='A', dtype=float)
    expected_B = pd.Series([2.0, 2.0, 4.0, 4.0, np.nan], index=sample_df_with_nans.index, name='B', dtype=float)
    assert_series_equal(df_bfill['A'], expected_A)
    assert_series_equal(df_bfill['B'], expected_B)

def test_handle_missing_values_dropna(sample_df_with_nans):
    df_dropna = data_processing.handle_missing_values(sample_df_with_nans.copy(), strategy='dropna')
    assert df_dropna.empty 

def test_handle_missing_values_mean(sample_df_with_nans):
    df_mean = data_processing.handle_missing_values(sample_df_with_nans.copy(), strategy='mean')
    mean_A = sample_df_with_nans['A'].mean() 
    mean_B = sample_df_with_nans['B'].mean() 
    expected_A = pd.Series([1.0, mean_A, 3.0, mean_A, 5.0], index=sample_df_with_nans.index, name='A',dtype=float)
    expected_B = pd.Series([mean_B, 2.0, mean_B, 4.0, mean_B], index=sample_df_with_nans.index, name='B',dtype=float)
    assert_series_equal(df_mean['A'], expected_A)
    assert_series_equal(df_mean['B'], expected_B)
    assert_series_equal(df_mean['C'], sample_df_with_nans['C']) 

def test_handle_missing_values_none_df():
    assert data_processing.handle_missing_values(None) is None

# Tests for calculate_daily_returns
def test_calculate_daily_returns_normal(sample_price_df):
    df_returns = data_processing.calculate_daily_returns(sample_price_df.copy(), price_column='close')
    assert 'daily_return' in df_returns.columns
    assert pd.isna(df_returns['daily_return'].iloc[0]) 
    assert df_returns['daily_return'].iloc[1] == pytest.approx(0.2) 
    assert df_returns['daily_return'].iloc[2] == pytest.approx((11.0-12.0)/12.0)

def test_calculate_daily_returns_with_nans(sample_price_df):
    df_copy = sample_price_df.copy()
    df_copy.loc[df_copy.index[2], 'close'] = np.nan 
    df_returns = data_processing.calculate_daily_returns(df_copy, price_column='close')
    assert pd.isna(df_returns['daily_return'].iloc[2]) 
    assert pd.isna(df_returns['daily_return'].iloc[3])

def test_calculate_daily_returns_with_zero_price(sample_price_df):
    df_copy = sample_price_df.copy()
    df_copy.loc[df_copy.index[1], 'close'] = 0.0 
    df_returns = data_processing.calculate_daily_returns(df_copy, price_column='close')
    assert df_returns['daily_return'].iloc[1] == -1.0
    assert pd.isna(df_returns['daily_return'].iloc[2])

def test_calculate_daily_returns_empty_df():
    assert data_processing.calculate_daily_returns(pd.DataFrame(columns=['date','close']), price_column='close').empty
    assert data_processing.calculate_daily_returns(None, price_column='close') is None

def test_calculate_daily_returns_missing_column(sample_price_df):
    df_result = data_processing.calculate_daily_returns(sample_price_df.copy(), price_column='non_existent')
    assert_frame_equal(df_result, sample_price_df) 

# Tests for calculate_moving_average
def test_calculate_moving_average_normal(sample_price_df):
    window = 3
    df_ma = data_processing.calculate_moving_average(sample_price_df.copy(), window=window, price_column='close')
    ma_col = f'MA_{window}'
    assert ma_col in df_ma.columns
    assert df_ma[ma_col].iloc[0] == pytest.approx(10.0) 
    assert df_ma[ma_col].iloc[1] == pytest.approx((10.0+12.0)/2) 
    assert df_ma[ma_col].iloc[2] == pytest.approx((10.0+12.0+11.0)/3) 
    assert df_ma[ma_col].iloc[3] == pytest.approx((12.0+11.0+13.0)/3) 

def test_calculate_moving_average_empty_df():
    assert data_processing.calculate_moving_average(pd.DataFrame(columns=['date','close']), window=5).empty
    assert data_processing.calculate_moving_average(None, window=5) is None

def test_calculate_moving_average_invalid_window(sample_price_df):
    df_result = data_processing.calculate_moving_average(sample_price_df.copy(), window=0) 
    assert_frame_equal(df_result, sample_price_df)

def test_calculate_moving_average_missing_column(sample_price_df):
    df_result = data_processing.calculate_moving_average(sample_price_df.copy(), window=5, price_column='non_existent')
    assert_frame_equal(df_result, sample_price_df)

# Tests for calculate_rsi
def test_calculate_rsi_structure(sample_price_df):
    window = 14
    df_rsi = data_processing.calculate_rsi(sample_price_df.copy(), window=window, price_column='close')
    rsi_col = f'RSI_{window}'
    assert rsi_col in df_rsi.columns
    assert pd.isna(df_rsi[rsi_col].iloc[0]) 
    for i in range(1, window): 
        val = df_rsi[rsi_col].iloc[i]
        assert pd.isna(val) or val == 0 or val == 100
    assert ((df_rsi[rsi_col].iloc[window:].dropna() >= 0) & (df_rsi[rsi_col].iloc[window:].dropna() <= 100)).all()

def test_calculate_rsi_known_values(rsi_fixture_sample_df):
    window = 14
    df_rsi = data_processing.calculate_rsi(rsi_fixture_sample_df.copy(), window=window, price_column='close')
    rsi_col = f'RSI_{window}'
    # For df.index[13] (14th day), RSI should be 0.0 due to fillna(0) for initial avg_gain/loss
    # if rolling mean doesn't have enough periods (which it does for the SMA part at index 13 of gain/loss).
    # The issue is how avg_gain/loss series (indexed from 0 after diff().dropna()) map to original df.
    # rsi.iloc[13] corresponds to original df.index[13].
    # avg_gain.iloc[12] and avg_loss.iloc[12] are used for df.index[13]'s RSI.
    # These are from rolling(window=14, min_periods=14).mean() which are NaN then fillna(0).
    assert df_rsi[rsi_col].loc[rsi_fixture_sample_df.index[13]] == pytest.approx(0.0, abs=0.01)
    # For df.index[14] (15th day), this is the first RSI based on a full 14-period SMA of gains/losses.
    assert df_rsi[rsi_col].loc[rsi_fixture_sample_df.index[14]] == pytest.approx(70.465, abs=0.01)
    # For df.index[15] (16th day), this is the first RSI using smoothed averages.
    assert df_rsi[rsi_col].loc[rsi_fixture_sample_df.index[15]] == pytest.approx(66.249, abs=0.01)


# Tests for calculate_bollinger_bands
def test_calculate_bollinger_bands_structure(sample_price_df):
    window = 20 
    num_std_dev = 2
    df_bb = data_processing.calculate_bollinger_bands(sample_price_df.copy(), window=window, num_std_dev=num_std_dev, price_column='close')
    
    mid_col = f'BB_Mid_{window}'
    upper_col = f'BB_Upper_{window}'
    lower_col = f'BB_Lower_{window}'
    
    assert mid_col in df_bb.columns
    assert upper_col in df_bb.columns
    assert lower_col in df_bb.columns
    
    expected_ma = sample_price_df['close'].rolling(window=window, min_periods=1).mean()
    assert_series_equal(df_bb[mid_col], expected_ma, check_names=False)

    std_dev_series = sample_price_df['close'].rolling(window=window, min_periods=1).std()
    # When std_dev_series is NaN (e.g. first element), the result of arithmetic with it is NaN.
    # This is the correct behavior for upper/lower bands.
    expected_upper = df_bb[mid_col] + (std_dev_series * num_std_dev)
    expected_lower = df_bb[mid_col] - (std_dev_series * num_std_dev)
    assert_series_equal(df_bb[upper_col], expected_upper, check_names=False)
    assert_series_equal(df_bb[lower_col], expected_lower, check_names=False)

def test_calculate_bollinger_bands_empty_df():
    assert data_processing.calculate_bollinger_bands(pd.DataFrame(columns=['date','close']), window=5).empty
    assert data_processing.calculate_bollinger_bands(None, window=5) is None

def test_calculate_bollinger_bands_invalid_params(sample_price_df):
    assert_frame_equal(data_processing.calculate_bollinger_bands(sample_price_df.copy(), window=0), sample_price_df)
    assert_frame_equal(data_processing.calculate_bollinger_bands(sample_price_df.copy(), window=5, num_std_dev=0), sample_price_df)

def test_calculate_bollinger_bands_missing_column(sample_price_df):
    df_result = data_processing.calculate_bollinger_bands(sample_price_df.copy(), window=5, price_column='non_existent')
    assert_frame_equal(df_result, sample_price_df)
