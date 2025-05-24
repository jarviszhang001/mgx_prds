import argparse
import os
import pandas as pd
import sys

# Adjust sys.path to allow imports from the 'src' directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# If main.py is in the project root, src_path should be 'src'.
# If main.py is in src/, then this needs adjustment or direct imports.
# Assuming main.py is in the project root 'akshare_quant_analysis_system/'
src_path = os.path.join(current_script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src import data_acquisition
    from src import data_processing
    from src import analysis_engine
    from src import reporting
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print(f"Please ensure 'main.py' is in the project root directory 'akshare_quant_analysis_system/'")
    print(f"and the 'src' directory with __init__.py exists and is populated.")
    sys.exit(1)

def handle_fetch(args):
    """Handles the 'fetch' command to acquire stock or index data."""
    print(f"Fetching data for code: {args.code}, type: {args.type}...")
    if args.type == 'stock':
        data_df = data_acquisition.fetch_stock_data(
            stock_code=args.code,
            start_date=args.start_date.replace("-", ""), # Convert YYYY-MM-DD to YYYYMMDD
            end_date=args.end_date.replace("-", "")
        )
    elif args.type == 'index':
        data_df = data_acquisition.fetch_index_data(
            index_code=args.code,
            start_date=args.start_date.replace("-", ""),
            end_date=args.end_date.replace("-", "")
        )
    else:
        print(f"Error: Invalid data type '{args.type}'. Must be 'stock' or 'index'.")
        return

    if data_df is not None and not data_df.empty:
        abs_output_file = os.path.join(current_script_dir, args.output_file)
        output_dir = os.path.dirname(abs_output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
            
        try:
            data_df.to_csv(abs_output_file, index=False)
            print(f"Data fetched successfully and saved to {abs_output_file} (relative: {args.output_file})")
        except Exception as e:
            print(f"Error saving data to {abs_output_file}: {e}")
    else:
        print(f"Failed to fetch data for {args.code}.")

def handle_analyze(args):
    """Handles the 'analyze' command to process data and generate signals."""
    abs_input_file = os.path.join(current_script_dir, args.input_file)
    abs_output_file = os.path.join(current_script_dir, args.output_file)
    print(f"Analyzing data from: {abs_input_file} (relative: {args.input_file})...")

    # Load data
    data_df = data_processing.load_data_from_csv(abs_input_file)
    if data_df is None:
        return # Error message handled by load_data_from_csv

    processed_df = data_df.copy()

    # Handle missing values (optional, could add CLI arg for strategy)
    processed_df = data_processing.handle_missing_values(processed_df, strategy='ffill')
    
    # Calculate indicators based on provided arguments
    active_signal_generators = []

    if args.sma_short and args.sma_long:
        print(f"Calculating SMA with short window {args.sma_short} and long window {args.sma_long}...")
        processed_df = data_processing.calculate_moving_average(processed_df, window=args.sma_short)
        processed_df = data_processing.calculate_moving_average(processed_df, window=args.sma_long)
        processed_df = analysis_engine.generate_sma_signals(processed_df, short_window=args.sma_short, long_window=args.sma_long)
        active_signal_generators.append('sma_signal')
        
    if args.rsi_window:
        print(f"Calculating RSI with window {args.rsi_window}...")
        processed_df = data_processing.calculate_rsi(processed_df, window=args.rsi_window)
        processed_df = analysis_engine.generate_rsi_signals(
            processed_df, 
            rsi_window=args.rsi_window, 
            oversold_threshold=args.rsi_oversold, 
            overbought_threshold=args.rsi_overbought
        )
        active_signal_generators.append('rsi_signal')

    if args.bb_window:
        print(f"Calculating Bollinger Bands with window {args.bb_window}...")
        processed_df = data_processing.calculate_bollinger_bands(
            processed_df, 
            window=args.bb_window, 
            num_std_dev=args.bb_std_dev
        )
        processed_df = analysis_engine.generate_bollinger_band_signals(
            processed_df, 
            bb_window=args.bb_window
        )
        active_signal_generators.append('bb_signal')

    if not active_signal_generators:
        print("No analysis strategies selected (e.g., SMA, RSI, BB). Saving data without new signals.")
    elif len(active_signal_generators) > 1:
        print(f"Combining signals ({', '.join(active_signal_generators)}) using '{args.combine_strategy}' strategy...")
        processed_df = analysis_engine.combine_signals(
            processed_df, 
            signal_columns=active_signal_generators, 
            strategy=args.combine_strategy
        )
    else:
        # If only one signal generator was used, its output is the 'combined_signal' effectively
        # Or, we can just use that specific signal column, e.g. 'sma_signal'
        print(f"Only one signal type ({active_signal_generators[0]}) generated. This will be the primary signal.")
        # No explicit combined_signal column is made if only one strategy is run unless combine_signals is called.
        # For consistency in 'report', it might expect 'combined_signal' or a user-specified one.

    # Ensure output directory exists
    output_dir = os.path.dirname(abs_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        processed_df.to_csv(abs_output_file, index=False)
        print(f"Analysis complete. Processed data saved to {abs_output_file} (relative: {args.output_file})")
    except Exception as e:
        print(f"Error saving analyzed data to {abs_output_file}: {e}")


def handle_report(args):
    """Handles the 'report' command to generate and display/save reports."""
    abs_input_file = os.path.join(current_script_dir, args.input_file)
    print(f"Generating report for: {abs_input_file} (relative: {args.input_file}) using signal column '{args.signal_column}'...")

    analyzed_df = data_processing.load_data_from_csv(abs_input_file)
    if analyzed_df is None:
        return

    if args.signal_column not in analyzed_df.columns:
        print(f"Error: Specified signal column '{args.signal_column}' not found in {abs_input_file}.")
        print(f"Available columns: {analyzed_df.columns.tolist()}")
        # Attempt to find a default or existing signal column if 'combined_signal' is missing
        potential_signals = [s for s in ['combined_signal', 'sma_signal', 'rsi_signal', 'bb_signal'] if s in analyzed_df.columns]
        if args.signal_column == 'combined_signal' and not potential_signals:
             print("No default signal columns ('combined_signal', 'sma_signal', etc.) found.")
             return
        elif args.signal_column == 'combined_signal' and potential_signals:
            args.signal_column = potential_signals[0] # Use the first available one
            print(f"Defaulting to use signal column: '{args.signal_column}' as 'combined_signal' was not found.")
        else: # User specified a column that is not found, and it wasn't the default 'combined_signal'
            return


    # Define paths for saving plots and reports
    input_basename = os.path.splitext(os.path.basename(abs_input_file))[0]
    
    # Paths should be relative to the project root for consistency in the report
    # docs/plots and docs/reports
    plot_dir_rel = os.path.join("docs", "plots")
    report_dir_rel = os.path.join("docs", "reports")

    abs_plot_dir = os.path.join(current_script_dir, plot_dir_rel)
    abs_report_dir = os.path.join(current_script_dir, report_dir_rel)

    os.makedirs(abs_plot_dir, exist_ok=True)
    os.makedirs(abs_report_dir, exist_ok=True)

    # Plot paths for saving (absolute)
    abs_price_signals_plot_path = os.path.join(abs_plot_dir, f"{input_basename}_price_signals.png")
    abs_equity_curve_plot_path = os.path.join(abs_plot_dir, f"{input_basename}_equity_curve.png")
    # Report path for saving (absolute)
    abs_report_file_path = os.path.join(abs_report_dir, f"{input_basename}_report.md")
    
    # Plot paths for embedding in MD report (should be relative to the report file)
    # Report is in docs/reports/, plots are in docs/plots/
    # So, path from report to plot is ../plots/filename.png
    rel_price_signals_plot_path_for_md = os.path.join("..", "plots", f"{input_basename}_price_signals.png")
    rel_equity_curve_plot_path_for_md = os.path.join("..", "plots", f"{input_basename}_equity_curve.png")
    
    # Generate plots
    ma_cols_present = [col for col in analyzed_df.columns if col.startswith('MA_')]
    reporting.plot_price_and_signals(
        analyzed_df,
        price_column=args.price_column,
        signal_column=args.signal_column,
        ma_columns=ma_cols_present,
        plot_title=f"Price & Signals for {input_basename} ({args.signal_column})",
        save_path=abs_price_signals_plot_path # Save to absolute path
    )

    # Generate performance summary
    performance_summary = reporting.generate_performance_summary(
        analyzed_df,
        price_column=args.price_column,
        signal_column=args.signal_column,
        initial_capital=args.initial_capital,
        risk_free_rate_annual=0.02 # Example, could be an arg
    )

    if performance_summary is None:
        print("Failed to generate performance summary.")
        return

    print("\n--- Performance Summary ---")
    print(performance_summary.drop('portfolio_value_series', errors='ignore'))

    # Plot equity curve
    abs_equity_curve_plot_path_used = None
    if 'portfolio_value_series' in performance_summary:
        reporting.plot_equity_curve(
            performance_summary['portfolio_value_series'],
            plot_title=f"Equity Curve for {input_basename} ({args.signal_column})",
            save_path=abs_equity_curve_plot_path # Save to absolute path
        )
        abs_equity_curve_plot_path_used = abs_equity_curve_plot_path # record that it was attempted
    
    # Save report to file
    plot_paths_for_md_report = {}
    if os.path.exists(abs_price_signals_plot_path):
         plot_paths_for_md_report["Price & Signals Plot"] = rel_price_signals_plot_path_for_md
    if abs_equity_curve_plot_path_used and os.path.exists(abs_equity_curve_plot_path_used):
         plot_paths_for_md_report["Equity Curve"] = rel_equity_curve_plot_path_for_md
        
    report_text_notes = (f"Report for {input_basename}, using signal: {args.signal_column}.\n"
                         f"Initial capital: {args.initial_capital}, Price column: {args.price_column}.")
    
    reporting.save_report_to_file(
        metrics=performance_summary,
        plot_paths=plot_paths_for_md_report, # Use relative paths for MD embedding
        report_text=report_text_notes,
        report_path=abs_report_file_path # Save report to absolute path
    )
    print(f"\nReport generation complete. Files saved in {abs_plot_dir} and {abs_report_dir}")


def main():
    """Main function to parse arguments and call handlers."""
    parser = argparse.ArgumentParser(description="Akshare Quantitative Analysis System CLI")
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # --- Fetch command ---
    fetch_parser = subparsers.add_parser("fetch", help="Fetch stock or index data.")
    fetch_parser.add_argument("--type", type=str, required=True, choices=['stock', 'index'], help="Type of data to fetch (stock or index).")
    fetch_parser.add_argument("--code", type=str, required=True, help="Stock or index code.")
    fetch_parser.add_argument("--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format.")
    fetch_parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD format.")
    fetch_parser.add_argument("--output_file", type=str, required=True, help="Path to save the fetched data (e.g., data/<code>.csv).")
    fetch_parser.set_defaults(func=handle_fetch)

    # --- Analyze command ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data and generate trading signals.")
    analyze_parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV data file.")
    analyze_parser.add_argument("--output_file", type=str, required=True, help="Path to save the analyzed data with signals.")
    
    analyze_parser.add_argument("--sma_short", type=int, help="Short window for SMA strategy.")
    analyze_parser.add_argument("--sma_long", type=int, help="Long window for SMA strategy.")
    
    analyze_parser.add_argument("--rsi_window", type=int, help="Window for RSI strategy.")
    analyze_parser.add_argument("--rsi_oversold", type=int, default=30, help="RSI oversold threshold (default: 30).")
    analyze_parser.add_argument("--rsi_overbought", type=int, default=70, help="RSI overbought threshold (default: 70).")
    
    analyze_parser.add_argument("--bb_window", type=int, help="Window for Bollinger Bands strategy.")
    analyze_parser.add_argument("--bb_std_dev", type=int, default=2, help="Standard deviations for Bollinger Bands (default: 2).")
    
    analyze_parser.add_argument("--combine_strategy", type=str, choices=['majority', 'unanimous'], default='majority', help="Strategy to combine signals if multiple indicators are used (default: majority).")
    analyze_parser.set_defaults(func=handle_analyze)

    # --- Report command ---
    report_parser = subparsers.add_parser("report", help="Generate reports and visualizations from analyzed data.")
    report_parser.add_argument("--input_file", type=str, required=True, help="Path to the analyzed CSV data file (with signals).")
    report_parser.add_argument("--signal_column", type=str, default='combined_signal', help="Name of the signal column to use for reporting (default: combined_signal).")
    report_parser.add_argument("--price_column", type=str, default='close', help="Name of the price column (default: close).")
    report_parser.add_argument("--initial_capital", type=float, default=100000.0, help="Initial capital for performance summary (default: 100000.0).")
    report_parser.set_defaults(func=handle_report)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
