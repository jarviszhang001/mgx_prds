import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure src directory (parent of current_dir) is in sys.path for module import in __main__
import sys
current_dir_for_sys_path = os.path.dirname(os.path.abspath(__file__))
project_root_for_sys_path = os.path.dirname(current_dir_for_sys_path)
if project_root_for_sys_path not in sys.path:
    sys.path.insert(0, project_root_for_sys_path)

from src import data_processing
from src import analysis_engine


def plot_price_and_signals(df: pd.DataFrame, 
                           price_column: str = 'close', 
                           signal_column: str = 'combined_signal', 
                           ma_columns: list[str] = None,
                           plot_title: str = "Price and Trading Signals",
                           save_path: str = None) -> None:
    """
    Plots the price series along with buy/sell signals and optional moving averages.

    Args:
        df (pd.DataFrame): DataFrame containing price data, signal column, and optionally MA columns.
                           Must have a 'date' column for the x-axis.
        price_column (str): Name of the column containing the price data to plot.
        signal_column (str): Name of the column containing trading signals (1 for Buy, -1 for Sell, 0 for Hold).
        ma_columns (list[str], optional): List of column names for moving averages to plot. Defaults to None.
        plot_title (str, optional): Title for the plot.
        save_path (str, optional): Path to save the plot image. If None, plot is displayed.
                                   Example: "docs/plots/price_signals.png"
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot generate plot.")
        return

    if 'date' not in df.columns:
        print("Error: 'date' column not found in DataFrame for plotting.")
        return
        
    if price_column not in df.columns:
        print(f"Error: Price column '{price_column}' not found.")
        return

    # Convert 'date' column to datetime if it's not already, for proper plotting
    df_plot = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_plot['date']):
        df_plot['date'] = pd.to_datetime(df_plot['date'])
    df_plot.sort_values('date', inplace=True)


    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid") # Using seaborn style
    
    # Plot price
    plt.plot(df_plot['date'], df_plot[price_column], label=price_column.capitalize(), color='blue', alpha=0.7)

    # Plot Moving Averages if provided
    if ma_columns:
        for ma_col in ma_columns:
            if ma_col in df_plot.columns:
                plt.plot(df_plot['date'], df_plot[ma_col], label=ma_col, alpha=0.7)
            else:
                print(f"Warning: MA column '{ma_col}' not found in DataFrame.")

    # Plot signals if signal column exists
    if signal_column in df_plot.columns:
        buy_signals = df_plot[df_plot[signal_column] == 1]
        sell_signals = df_plot[df_plot[signal_column] == -1]

        plt.scatter(buy_signals['date'], buy_signals[price_column], 
                    label='Buy Signal', marker='^', color='green', s=100, alpha=1, zorder=5)
        plt.scatter(sell_signals['date'], sell_signals[price_column], 
                    label='Sell Signal', marker='v', color='red', s=100, alpha=1, zorder=5)
    else:
        print(f"Warning: Signal column '{signal_column}' not provided or not found. Signals will not be plotted.")

    plt.title(plot_title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300) # Save with higher resolution
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
    else:
        plt.show()
    plt.close() # Close the figure to free memory

def generate_performance_summary(df: pd.DataFrame, 
                                 price_column: str = 'close', 
                                 signal_column: str = 'combined_signal', 
                                 initial_capital: float = 100000.0,
                                 risk_free_rate_annual: float = 0.0) -> pd.Series | None:
    """
    Simulates a simple backtest and calculates key performance metrics.

    Args:
        df (pd.DataFrame): DataFrame with 'date', price_column, and signal_column.
                           Signals: 1 for Buy, -1 for Sell, 0 for Hold.
                           Assumes trades occur at the price_column value on the signal day.
        price_column (str): Column name for the trading price (e.g., 'close').
        signal_column (str): Column name for trading signals.
        initial_capital (float): Starting capital for the backtest.
        risk_free_rate_annual (float): Annual risk-free rate for Sharpe ratio calculation (e.g., 0.02 for 2%).

    Returns:
        pd.Series | None: A pandas Series containing performance metrics, or None if errors occur.
                          Metrics include: 'Total Return (%)', 'Annualized Return (%)', 
                          'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Number of Trades',
                          'Final Portfolio Value', 'Peak Portfolio Value', 'Lowest Portfolio Value (Drawdown)',
                          'portfolio_value_series' (pd.Series of portfolio values over time).
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot generate performance summary.")
        return None
    
    required_cols = ['date', price_column, signal_column]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in DataFrame.")
            return None

    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date to ensure correct backtesting logic
    data.sort_values('date', inplace=True)
    data.set_index('date', inplace=True, drop=False) # Keep date column

    capital = initial_capital
    position_shares = 0  # Number of shares held
    portfolio_values = [] # Will store portfolio value at each step
    trade_log = [] # To store {'price_in': float, 'price_out': float} for win rate

    last_buy_price = None

    for i in range(len(data)):
        signal = data[signal_column].iloc[i]
        price = data[price_column].iloc[i]

        if pd.isna(signal) or pd.isna(price):
            current_value = capital + (position_shares * (data[price_column].iloc[i-1] if i > 0 and not pd.isna(data[price_column].iloc[i-1]) else 0) )
            portfolio_values.append(current_value if i > 0 else initial_capital)
            continue

        if signal == 1: # Buy signal
            if position_shares == 0: # If not holding, buy
                position_shares = capital / price 
                capital = 0
                last_buy_price = price # Log buy price
        elif signal == -1: # Sell signal
            if position_shares > 0: # If holding, sell all
                capital += position_shares * price
                position_shares = 0
                if last_buy_price is not None: # A buy must have preceded this sell
                    trade_log.append({'price_in': last_buy_price, 'price_out': price})
                    last_buy_price = None # Reset for next buy

        current_portfolio_value = capital + (position_shares * price)
        portfolio_values.append(current_portfolio_value)

    data['portfolio_value'] = portfolio_values

    # --- Calculate Metrics ---
    if data['portfolio_value'].empty:
        print("Warning: Portfolio value series is empty after backtest. Cannot calculate metrics.")
        # Return a Series of NaNs or zeros for expected metrics
        metrics_cols = ["Total Return (%)", "Annualized Return (%)", "Sharpe Ratio", "Max Drawdown (%)", 
                        "Win Rate (%)", "Number of Trades", "Initial Portfolio Value", "Final Portfolio Value", 
                        "Peak Portfolio Value", "Lowest Portfolio Value (Drawdown)"]
        empty_metrics = pd.Series(data=[0.0]*len(metrics_cols), index=metrics_cols)
        empty_metrics["portfolio_value_series"] = pd.Series([initial_capital], index=df.index[[0]] if not df.index.empty else None)
        return empty_metrics

    final_value = data['portfolio_value'].iloc[-1]
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100

    # Annualized Return
    time_delta = data.index[-1] - data.index[0]
    days = time_delta.days
    years = days / 365.25 if days > 0 else 0 # Avoid division by zero if timespan is too short
    
    annualized_return_pct = 0.0
    if years > 0 and initial_capital > 0: # Ensure years and initial_capital are positive
        annualized_return_pct = (((final_value / initial_capital) ** (1 / years)) - 1) * 100
    
    # Sharpe Ratio
    data['daily_portfolio_return'] = data['portfolio_value'].pct_change().fillna(0)
    sharpe_ratio = 0.0
    # Ensure there are returns to calculate std dev and it's not zero
    if not data['daily_portfolio_return'].empty and data['daily_portfolio_return'].std() != 0 and days > 0:
        # Assuming 252 trading days in a year
        daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/252) - 1 if risk_free_rate_annual !=0 else 0.0
        excess_returns = data['daily_portfolio_return'] - daily_risk_free_rate
        # Check if excess_returns std is zero (can happen if all returns are same, e.g., all zero)
        if excess_returns.std() != 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else: # if std is zero but mean is not, sharpe can be inf. If mean is also zero, it's NaN or 0.
            sharpe_ratio = np.inf if excess_returns.mean() > 0 else (-np.inf if excess_returns.mean() < 0 else 0.0)


    # Max Drawdown
    data['cumulative_max'] = data['portfolio_value'].cummax()
    data['drawdown'] = (data['portfolio_value'] - data['cumulative_max']) / data['cumulative_max']
    max_drawdown_pct = data['drawdown'].min() * 100 if not data['drawdown'].empty else 0.0
    peak_portfolio_value = data['cumulative_max'].max() if not data['cumulative_max'].empty else initial_capital
    lowest_portfolio_value_during_drawdown = data['portfolio_value'].loc[data['drawdown'].idxmin()] if not data['drawdown'].empty and not data['drawdown'].isnull().all() else initial_capital


    # Win Rate & Number of Trades
    num_trades = len(trade_log)
    profitable_trades = 0
    if num_trades > 0:
        for trade in trade_log:
            if trade['price_out'] > trade['price_in']:
                profitable_trades += 1
        win_rate_pct = (profitable_trades / num_trades) * 100 if num_trades > 0 else 0.0
    else:
        win_rate_pct = 0.0

    metrics = pd.Series({
        "Total Return (%)": total_return_pct,
        "Annualized Return (%)": annualized_return_pct,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown_pct,
        "Win Rate (%)": win_rate_pct,
        "Number of Trades": num_trades,
        "Initial Portfolio Value": initial_capital,
        "Final Portfolio Value": final_value,
        "Peak Portfolio Value": peak_portfolio_value,
        "Lowest Portfolio Value (Drawdown)": lowest_portfolio_value_during_drawdown,
        "portfolio_value_series": data['portfolio_value'] # For equity curve plot
    })
    
    print("\nPerformance Summary Generated:")
    print(metrics.drop('portfolio_value_series', errors='ignore')) # Don't print the whole series here
    return metrics

def plot_equity_curve(portfolio_values_series: pd.Series, 
                      plot_title: str = "Equity Curve", 
                      save_path: str = None) -> None:
    """
    Plots the equity curve (portfolio value over time).

    Args:
        portfolio_values_series (pd.Series): Series containing portfolio values, with a DatetimeIndex.
        plot_title (str, optional): Title for the plot.
        save_path (str, optional): Path to save the plot image. If None, plot is displayed.
    """
    if portfolio_values_series is None or portfolio_values_series.empty:
        print("Portfolio value series is None or empty. Cannot plot equity curve.")
        return

    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    portfolio_values_series.plot(label='Portfolio Value', color='navy')
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Equity curve plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving equity curve plot to {save_path}: {e}")
    else:
        plt.show()
    plt.close()

def save_report_to_file(metrics: pd.Series, 
                        plot_paths: dict = None, 
                        report_text: str = None, 
                        report_path: str = "docs/reports/performance_report.md") -> None:
    """
    Saves the performance metrics, paths to plots, and additional report text to a Markdown file.

    Args:
        metrics (pd.Series): Series containing performance metrics.
        plot_paths (dict, optional): Dictionary of plot names to their save paths 
                                     (e.g., {"Price & Signals Plot": "path/to/plot.png"}).
        report_text (str, optional): Additional text or context to include in the report.
        report_path (str): Path to save the report file.
    """
    if metrics is None:
        print("Metrics are None. Cannot save report.")
        return

    try:
        # Ensure the directory for the report exists
        report_dir = os.path.dirname(report_path)
        if report_dir: # Check if report_dir is not an empty string (e.g. if path is just "file.md")
            os.makedirs(report_dir, exist_ok=True)
            
        with open(report_path, 'w') as f:
            f.write(f"# Performance Report\n\n")
            f.write(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if report_text:
                f.write(f"## Notes\n{report_text}\n\n")
            
            f.write(f"## Key Performance Indicators (KPIs)\n\n")
            metrics_to_save = metrics.drop('portfolio_value_series', errors='ignore') # Exclude the raw series
            for idx, val in metrics_to_save.items():
                if isinstance(val, float):
                    f.write(f"- **{idx}**: {val:.2f}\n")
                else:
                    f.write(f"- **{idx}**: {val}\n")
            
            if plot_paths:
                f.write(f"\n## Visualizations\n\n")
                for plot_name, path in plot_paths.items():
                    # Make path relative to the report file if possible, for portability
                    try:
                        relative_plot_path = os.path.relpath(path, start=report_dir if report_dir else ".")
                    except ValueError: # Happens if paths are on different drives (Windows)
                        relative_plot_path = path 
                    f.write(f"### {plot_name}\n")
                    f.write(f"![{plot_name}]({relative_plot_path})\n\n")

        print(f"Performance report saved to {report_path}")
    except Exception as e:
        print(f"Error saving report to {report_path}: {e}")

if __name__ == '__main__':
    print("--- Reporting Module Example ---")

    # Create a sample DataFrame (similar to analysis_engine.py for consistency)
    date_rng = pd.date_range(start='2023-01-01', end='2023-06-01', freq='B') # Extended for more data
    sample_size = len(date_rng)
    np.random.seed(42) # For reproducibility
    close_prices = 100 + np.random.randn(sample_size).cumsum()
    
    sample_df = pd.DataFrame({
        'date': date_rng,
        'close': close_prices,
        'open': close_prices - np.random.rand(sample_size) * 0.5,
        'high': close_prices + np.random.rand(sample_size) * 0.5,
        'low': close_prices - np.random.rand(sample_size) * 0.6,
        'volume': np.random.randint(1000, 5000, size=sample_size)
    })
    sample_df['low'] = sample_df[['low', 'open', 'close', 'high']].min(axis=1)
    sample_df['high'] = sample_df[['high', 'open', 'close']].max(axis=1)

    print("\nSample DataFrame created (first 5 rows):")
    print(sample_df.head())

    # 1. Process data
    df_processed = sample_df.copy()
    short_sma_window = 10 
    long_sma_window = 30  
    rsi_calc_window = 14
    bb_calc_window = 20

    df_processed = data_processing.calculate_moving_average(df_processed, window=short_sma_window)
    df_processed = data_processing.calculate_moving_average(df_processed, window=long_sma_window)
    df_processed = data_processing.calculate_rsi(df_processed, window=rsi_calc_window)
    df_processed = data_processing.calculate_bollinger_bands(df_processed, window=bb_calc_window)
    df_processed = data_processing.calculate_daily_returns(df_processed) # For some performance metrics
    
    print("\nDataFrame after adding indicators (tail):")
    print(df_processed.tail())

    # 2. Generate signals
    df_signals = df_processed.copy()
    df_signals = analysis_engine.generate_sma_signals(df_signals, short_window=short_sma_window, long_window=long_sma_window)
    df_signals = analysis_engine.generate_rsi_signals(df_signals, rsi_window=rsi_calc_window)
    df_signals = analysis_engine.generate_bollinger_band_signals(df_signals, bb_window=bb_calc_window)
    
    signal_cols = ['sma_signal', 'rsi_signal', 'bb_signal']
    df_signals = analysis_engine.combine_signals(df_signals, signal_columns=signal_cols, strategy='majority')

    print("\nDataFrame with combined signals (showing some signals):")
    print(df_signals[df_signals['combined_signal'] != 0].head())

    # 3. Reporting
    # Define plot save directory (relative to project root)
    # docs/plots/
    plot_save_dir = os.path.join(project_root_for_sys_path, "docs", "plots")
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
        print(f"Created directory: {plot_save_dir}")

    # Report save directory
    report_save_dir = os.path.join(project_root_for_sys_path, "docs", "reports")
    if not os.path.exists(report_save_dir):
        os.makedirs(report_save_dir)
        print(f"Created directory: {report_save_dir}")


    # Plot price and signals
    ma_cols_to_plot = [f'MA_{short_sma_window}', f'MA_{long_sma_window}']
    price_signals_plot_path = os.path.join(plot_save_dir, "price_and_combined_signals_example.png")
    print(f"\nPlotting price with combined signals, saving to: {price_signals_plot_path}")
    plot_price_and_signals(df_signals, 
                           price_column='close', 
                           signal_column='combined_signal', 
                           ma_columns=ma_cols_to_plot,
                           plot_title="Stock Price with Combined Signals and MAs (Example)",
                           save_path=price_signals_plot_path)
    
    # Generate performance summary
    print("\nGenerating performance summary for 'combined_signal'...")
    performance_metrics = generate_performance_summary(df_signals, 
                                                       price_column='close', 
                                                       signal_column='combined_signal',
                                                       initial_capital=100000,
                                                       risk_free_rate_annual=0.02)

    # Plot equity curve if metrics and series are generated
    equity_curve_plot_path = None
    if performance_metrics is not None and 'portfolio_value_series' in performance_metrics:
        equity_curve_plot_path = os.path.join(plot_save_dir, "equity_curve_example.png")
        print(f"\nPlotting equity curve, saving to: {equity_curve_plot_path}")
        plot_equity_curve(performance_metrics['portfolio_value_series'], 
                          plot_title="Portfolio Equity Curve (Example)",
                          save_path=equity_curve_plot_path)
    else:
        print("\nCould not generate performance metrics or portfolio value series, skipping equity curve plot.")

    # Save report to file
    if performance_metrics is not None:
        report_file_path = os.path.join(report_save_dir, "performance_report_example.md")
        report_notes = "This is an example performance report generated using sample data and a majority-vote combined signal strategy."
        
        plot_paths_for_report = {}
        if os.path.exists(price_signals_plot_path): # Check if plot was actually saved
             plot_paths_for_report["Price & Signals Plot"] = price_signals_plot_path
        if equity_curve_plot_path and os.path.exists(equity_curve_plot_path):
             plot_paths_for_report["Equity Curve"] = equity_curve_plot_path
        
        print(f"\nSaving performance report to: {report_file_path}")
        save_report_to_file(metrics=performance_metrics, 
                            plot_paths=plot_paths_for_report,
                            report_text=report_notes,
                            report_path=report_file_path)
    else:
        print("\nCould not generate performance metrics, skipping report saving.")

    print("\n--- End of Reporting Module Example ---")
