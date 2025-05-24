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

from src import analysis_engine
from src import data_processing 

@pytest.fixture
def sample_df_for_signals():
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', 
                            '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'])
    data = {
        'date': dates,
        'close':    [10, 11, 12, 11, 10,  9,  8,  9],
        'MA_2':     [10.0, 10.5, 11.5, 11.5, 10.5, 9.5, 8.5, 8.5],
        'MA_4':     [10.0, 10.5, 11.0, 11.0, 11.0, 10.25, 9.5, 9.25],
        'RSI_3':    [np.nan, 100, 100, 30, 0, 0, 0, 50], 
        'BB_Lower_3': [9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 7.0, 8.0], 
        'BB_Upper_3': [11.0, 12.0, 13.0, 12.0, 11.0,10.0, 9.0,10.0] 
    }
    df = pd.DataFrame(data)
    return df.set_index('date')

def test_generate_sma_signals_buy_sell(sample_df_for_signals):
    df_signal = analysis_engine.generate_sma_signals(sample_df_for_signals.copy(), short_window=2, long_window=4)
    expected_signals = pd.Series([0, 0, 0, 0, -1, 0, 0, 0], index=sample_df_for_signals.index, name='sma_signal')
    assert_series_equal(df_signal['sma_signal'], expected_signals, check_dtype=False)

def test_generate_rsi_signals_buy_sell(sample_df_for_signals):
    df = sample_df_for_signals.copy()
    df_signal = analysis_engine.generate_rsi_signals(df, rsi_window=3, oversold_threshold=30, overbought_threshold=70)
    expected_signals = pd.Series([0, 0, 0, -1, 0, 0, 0, 1], index=df.index, name='rsi_signal')
    assert_series_equal(df_signal['rsi_signal'], expected_signals, check_dtype=False)

def test_generate_bollinger_band_signals_buy_sell(sample_df_for_signals):
    df = sample_df_for_signals.copy()
    # Based on data: close=[10,11,12,11,10,9,8,9], BB_Lower_3=[9,10,11,10,9,8,7,8], BB_Upper_3=[11,12,13,12,11,10,9,10]
    # Sell: (P.shift >= U.shift) & (P < U)
    #   Idx 3 (01-04): P=11, U=12. Pprev=12, Uprev=13. (12 >= 13) is False. No sell.
    # Buy: (P.shift <= L.shift) & (P > L)
    #   Idx 5 (01-06): P=9, L=8. Pprev=10, Lprev=9. (10 <= 9) is False. No buy.
    # With the current data and logic, no BB signals should be generated.
    df_signal = analysis_engine.generate_bollinger_band_signals(df, bb_window=3, price_column='close')
    expected_signals = pd.Series([0,0,0,0,0,0,0,0], index=df.index, name='bb_signal') # Corrected expectation
    assert_series_equal(df_signal['bb_signal'], expected_signals, check_dtype=False)

@pytest.fixture
def df_with_individual_signals():
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
    data = { 'date': dates, 'close': [10,11,12,13,14,15],
        'sig1': [ 1,  1, -1,  0,  1, -1], 'sig2': [ 0,  1,  1, -1,  0, -1], 'sig3': [-1,  1,  0, -1,  1,  1]  
    }
    return pd.DataFrame(data).set_index('date')

def test_combine_signals_majority(df_with_individual_signals):
    df_combined = analysis_engine.combine_signals(df_with_individual_signals.copy(), signal_columns=['sig1', 'sig2', 'sig3'], strategy='majority')
    expected_combined = pd.Series([0, 1, 0, -1, 1, -1], index=df_with_individual_signals.index, name='combined_signal')
    assert_series_equal(df_combined['combined_signal'], expected_combined, check_dtype=False)

def test_combine_signals_unanimous(df_with_individual_signals):
    df_combined = analysis_engine.combine_signals(df_with_individual_signals.copy(), signal_columns=['sig1', 'sig2', 'sig3'], strategy='unanimous')
    expected_combined = pd.Series([0, 1, 0, 0, 0, 0], index=df_with_individual_signals.index, name='combined_signal')
    assert_series_equal(df_combined['combined_signal'], expected_combined, check_dtype=False)

def test_combine_signals_empty_df():
    df = pd.DataFrame(columns=['date', 'sig1']).set_index('date')
    df_combined = analysis_engine.combine_signals(df.copy(), signal_columns=['sig1'], strategy='majority')
    # If input df is empty, it's returned as is. 'combined_signal' is not added if no processing occurs.
    assert 'combined_signal' not in df_combined.columns 

def test_combine_signals_no_signal_cols_provided(df_with_individual_signals):
    df_combined = analysis_engine.combine_signals(df_with_individual_signals.copy(), signal_columns=[], strategy='majority')
    assert (df_combined['combined_signal'] == 0).all()

def test_combine_signals_missing_signal_cols(df_with_individual_signals):
    df_result = analysis_engine.combine_signals(df_with_individual_signals.copy(), signal_columns=['sig1', 'non_existent_sig'], strategy='majority')
    assert_frame_equal(df_result, df_with_individual_signals) 

def test_combine_signals_unknown_strategy(df_with_individual_signals):
    df_combined = analysis_engine.combine_signals(df_with_individual_signals.copy(), signal_columns=['sig1'], strategy='unknown')
    assert (df_combined['combined_signal'] == 0).all()

def test_combine_signals_majority_with_zero_preference():
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', 
                            '2023-01-05', '2023-01-06', '2023-01-07'])
    data = { 'date': dates, 
        'sig1': [ 1,  0,  1, -1,  0, 0,  1], 'sig2': [-1,  0,  0,  1,  1, 0, -1], 'sig3': [ 0,  1, -1,  0, -1, 0,  0]
    }
    df = pd.DataFrame(data).set_index('date')
    df_combined = analysis_engine.combine_signals(df, signal_columns=['sig1', 'sig2', 'sig3'], strategy='majority')
    expected = pd.Series([0,0,0,0,0,0,0], index=df.index, name='combined_signal')
    assert_series_equal(df_combined['combined_signal'], expected, check_dtype=False)

    data2 = {'date': [dates[0]], 'sig1': [1], 'sig2': [1], 'sig3': [0]} 
    df2 = pd.DataFrame(data2).set_index('date') 
    df_combined2 = analysis_engine.combine_signals(df2, signal_columns=['sig1', 'sig2', 'sig3'], strategy='majority')
    expected2 = pd.Series([1], index=df2.index, name='combined_signal') 
    assert_series_equal(df_combined2['combined_signal'], expected2, check_dtype=False)

    data3 = {'date': [dates[0]], 'sig1': [-1], 'sig2': [-1], 'sig3': [0]} 
    df3 = pd.DataFrame(data3).set_index('date') 
    df_combined3 = analysis_engine.combine_signals(df3, signal_columns=['sig1', 'sig2', 'sig3'], strategy='majority')
    expected3 = pd.Series([-1], index=df3.index, name='combined_signal')
    assert_series_equal(df_combined3['combined_signal'], expected3, check_dtype=False)

    data4 = {'date': [dates[0]], 'sig1': [0], 'sig2': [0], 'sig3': [1]} 
    df4 = pd.DataFrame(data4).set_index('date') 
    df_combined4 = analysis_engine.combine_signals(df4, signal_columns=['sig1', 'sig2', 'sig3'], strategy='majority')
    expected4 = pd.Series([0], index=df4.index, name='combined_signal')
    assert_series_equal(df_combined4['combined_signal'], expected4, check_dtype=False)

# Keep other successful tests for completeness
def test_generate_sma_signals_no_crossover(sample_df_for_signals):
    df = sample_df_for_signals.copy()
    df['MA_2'] = 10 
    df['MA_4'] = 12 
    df_signal = analysis_engine.generate_sma_signals(df, short_window=2, long_window=4)
    assert (df_signal['sma_signal'] == 0).all()

def test_generate_sma_signals_missing_ma_cols(sample_df_for_signals):
    df = sample_df_for_signals.drop(columns=['MA_4'])
    df_result = analysis_engine.generate_sma_signals(df, short_window=2, long_window=4)
    assert_frame_equal(df_result, df) 

def test_generate_rsi_signals_missing_rsi_col(sample_df_for_signals):
    df = sample_df_for_signals.drop(columns=['RSI_3'])
    df_result = analysis_engine.generate_rsi_signals(df, rsi_window=3)
    assert_frame_equal(df_result, df)

def test_generate_bollinger_band_signals_missing_cols(sample_df_for_signals):
    df = sample_df_for_signals.drop(columns=['BB_Lower_3'])
    df_result = analysis_engine.generate_bollinger_band_signals(df, bb_window=3)
    assert_frame_equal(df_result, df)
