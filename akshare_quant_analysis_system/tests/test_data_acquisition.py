import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # This should be akshare_quant_analysis_system/
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src import data_acquisition
# Assuming data_processing.load_data_from_csv will be tested here due to close relation with save_data_to_csv
from src import data_processing 

@pytest.fixture
def sample_stock_df():
    """Returns a sample DataFrame similar to what akshare.stock_zh_a_hist might return."""
    data = {
        '日期': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        '开盘': [10.0, 10.2, 10.1],
        '收盘': [10.2, 10.1, 10.3],
        '最高': [10.3, 10.3, 10.4],
        '最低': [9.9, 10.0, 10.0],
        '成交量': [1000, 1200, 1100],
        '成交额': [10000, 12200, 11300],
        '振幅': [0.04, 0.03, 0.04],
        '涨跌幅': [0.02, -0.01, 0.02],
        '涨跌额': [0.2, -0.1, 0.2],
        '换手率': [0.01, 0.012, 0.011]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_index_df():
    """Returns a sample DataFrame similar to what akshare.index_zh_a_hist might return."""
    data = {
        '日期': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        '开盘': [3000.0, 3010.0, 3005.0],
        '收盘': [3010.0, 3005.0, 3020.0],
        '最高': [3020.0, 3015.0, 3025.0],
        '最低': [2990.0, 3000.0, 3000.0],
        '成交量': [100000, 120000, 110000], # Note: '成交量' for index_zh_a_hist is in '手' (lots)
        '成交额': [1000000, 1220000, 1130000] # Note: '成交额' is in '元'
        # index_zh_a_hist also has '振幅', '涨跌幅', '涨跌额', '换手率'
    }
    return pd.DataFrame(data)

# Tests for fetch_stock_data
@patch('akshare.stock_zh_a_hist')
def test_fetch_stock_data_success(mock_ak_stock_hist, sample_stock_df):
    """Test successful fetching of stock data."""
    mock_ak_stock_hist.return_value = sample_stock_df.copy()
    
    df = data_acquisition.fetch_stock_data("600519", "20230101", "20230103")
    
    assert df is not None
    assert not df.empty
    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in expected_cols)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    mock_ak_stock_hist.assert_called_once_with(
        symbol="600519", period="daily", start_date="20230101", end_date="20230103", adjust="qfq"
    )

@patch('akshare.stock_zh_a_hist')
def test_fetch_stock_data_api_error(mock_ak_stock_hist):
    """Test handling of API error when fetching stock data."""
    mock_ak_stock_hist.side_effect = Exception("API network error")
    
    df = data_acquisition.fetch_stock_data("600519", "20230101", "20230103")
    
    assert df is None

@patch('akshare.stock_zh_a_hist')
def test_fetch_stock_data_empty_df(mock_ak_stock_hist):
    """Test handling of empty DataFrame returned by API for stock data."""
    mock_ak_stock_hist.return_value = pd.DataFrame() # Empty df
    
    df = data_acquisition.fetch_stock_data("000000", "20230101", "20230103")
    
    assert df is None

# Tests for fetch_index_data
@patch('akshare.index_zh_a_hist')
def test_fetch_index_data_success(mock_ak_index_hist, sample_index_df):
    """Test successful fetching of index data."""
    mock_ak_index_hist.return_value = sample_index_df.copy()
    
    df = data_acquisition.fetch_index_data("sh000001", "20230101", "20230103")
    
    assert df is not None
    assert not df.empty
    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    assert all(col in df.columns for col in expected_cols)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    mock_ak_index_hist.assert_called_once_with(
        symbol="000001", period="daily", start_date="20230101", end_date="20230103"
    )

@patch('akshare.index_zh_a_hist')
def test_fetch_index_data_api_error(mock_ak_index_hist):
    """Test handling of API error when fetching index data."""
    mock_ak_index_hist.side_effect = Exception("API network error")
    
    df = data_acquisition.fetch_index_data("sh000001", "20230101", "20230103")
    
    assert df is None

@patch('akshare.index_zh_a_hist')
def test_fetch_index_data_empty_df(mock_ak_index_hist):
    """Test handling of empty DataFrame returned by API for index data."""
    mock_ak_index_hist.return_value = pd.DataFrame() # Empty df
    
    df = data_acquisition.fetch_index_data("sz000000", "20230101", "20230103") # Non-existent index
    
    assert df is None


# Test for save_data_to_csv and data_processing.load_data_from_csv
def test_save_and_load_csv(tmp_path, sample_stock_df):
    """Test saving a DataFrame to CSV and loading it back."""
    # Modify sample_stock_df to match the structure after fetch_stock_data processing
    # (English column names, specific columns)
    processed_sample_df = sample_stock_df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume"
    })[['date', 'open', 'high', 'low', 'close', 'volume']]
    processed_sample_df['date'] = pd.to_datetime(processed_sample_df['date'])


    # Need to save relative to where data_acquisition.py expects it (../data from src/)
    # or provide an absolute path. tmp_path provides an absolute path.
    # The save_data_to_csv function in data_acquisition.py saves relative to its own location.
    # Let's test its behavior directly.
    
    # Since save_data_to_csv saves to directory "../data" relative to its own path (src/data_acquisition.py)
    # We need to mock os.path.abspath and os.makedirs if we don't want it to actually try to write there.
    # For an integration test like this, it's easier to let it write to tmp_path.
    # We can achieve this by temporarily changing the working directory or patching where it writes.

    # Alternative: Use a temporary directory that acts like the project root for the test.
    test_project_root = tmp_path
    src_dir = test_project_root / "src"
    data_dir = test_project_root / "data"
    src_dir.mkdir()
    data_dir.mkdir()

    # The file will be saved to tmp_path / "data" / "test_data.csv"
    # by save_data_to_csv if it thinks it's in src_dir.
    
    file_name = "test_save_load.csv"
    
    # Patch os.path.abspath to make save_data_to_csv use tmp_path structure
    # This means __file__ inside data_acquisition.py will resolve to a path inside src_dir (tmp)
    with patch('src.data_acquisition.os.path.abspath') as mock_abspath:
        mock_abspath.return_value = str(src_dir / "data_acquisition.py") # Mock its location
        
        save_success = data_acquisition.save_data_to_csv(processed_sample_df, file_name, directory="../data")
    
    assert save_success
    
    saved_file_path = data_dir / file_name
    assert saved_file_path.exists()

    # Test loading using data_processing.load_data_from_csv
    # load_data_from_csv also has path logic; pass the absolute path for simplicity here.
    loaded_df = data_processing.load_data_from_csv(str(saved_file_path))
    
    assert loaded_df is not None
    assert not loaded_df.empty
    # Convert date column back to datetime for proper comparison, as CSV load might make it string
    loaded_df['date'] = pd.to_datetime(loaded_df['date'])
    
    pd.testing.assert_frame_equal(processed_sample_df, loaded_df, check_dtype=True)

def test_save_data_to_csv_empty_df(tmp_path):
    """Test saving an empty DataFrame."""
    empty_df = pd.DataFrame()
    test_project_root = tmp_path
    src_dir = test_project_root / "src"
    src_dir.mkdir()

    with patch('src.data_acquisition.os.path.abspath') as mock_abspath:
        mock_abspath.return_value = str(src_dir / "data_acquisition.py")
        save_success = data_acquisition.save_data_to_csv(empty_df, "empty.csv", directory="../data")
    
    assert not save_success # Should return False for empty df

def test_save_data_to_csv_none_df(tmp_path):
    """Test saving a None DataFrame."""
    test_project_root = tmp_path
    src_dir = test_project_root / "src"
    src_dir.mkdir()
    with patch('src.data_acquisition.os.path.abspath') as mock_abspath:
        mock_abspath.return_value = str(src_dir / "data_acquisition.py")
        save_success = data_acquisition.save_data_to_csv(None, "none.csv", directory="../data")
    assert not save_success # Should return False for None df

def test_load_data_from_csv_file_not_found():
    """Test loading a non-existent CSV file."""
    df = data_processing.load_data_from_csv("non_existent_file.csv")
    assert df is None

def test_load_data_from_csv_empty_file(tmp_path):
    """Test loading an empty CSV file."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("") # Create an empty file
    df = data_processing.load_data_from_csv(str(empty_file))
    assert df is None # pandas read_csv on empty file raises EmptyDataError
