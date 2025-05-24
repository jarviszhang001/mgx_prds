# Akshare Quant Analysis System

## Description

Akshare Quant Analysis System is a Python-based system for quantitative financial analysis. It leverages the Akshare library for acquiring financial data (stocks and indices) and implements common technical indicators for generating trading signals. The system provides a Command Line Interface (CLI) for easy interaction, along with reporting capabilities to evaluate strategy performance.

## Features

*   **Data Acquisition:** Fetch historical stock and index data using Akshare.
*   **Data Processing:**
    *   Handle missing data.
    *   Calculate daily returns.
    *   Calculate technical indicators:
        *   Simple Moving Averages (SMA)
        *   Relative Strength Index (RSI)
        *   Bollinger Bands (BB)
*   **Signal Generation:**
    *   Generate trading signals based on SMA crossovers, RSI thresholds, and Bollinger Band breakouts.
    *   Combine multiple signals using 'majority' vote or 'unanimous' agreement strategies.
*   **Reporting & Performance:**
    *   Generate plots for price data, indicators, and trading signals.
    *   Calculate performance metrics (Total Return, Annualized Return, Sharpe Ratio, Max Drawdown, Win Rate, etc.).
    *   Plot equity curves.
    *   Save performance reports to Markdown files.
*   **Command Line Interface (CLI):**
    *   Easy-to-use CLI for fetching data, running analysis, and generating reports.
*   **Modular Design:** Code is organized into modules for data acquisition, processing, analysis, and reporting.
*   **Unit Tested:** Core functionalities are covered by unit tests.

## Project Structure

The project is organized as follows:

```
akshare_quant_analysis_system/
├── data/                     # Default directory for storing fetched and processed data (CSV files)
├── docs/                     # Documentation files
│   ├── design/               # System design documents (Markdown, Mermaid diagrams)
│   ├── plots/                # Saved plots from the reporting module
│   └── reports/              # Saved performance reports
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_acquisition.py   # Data fetching logic
│   ├── data_processing.py    # Data cleaning and indicator calculation
│   ├── analysis_engine.py    # Signal generation and strategy logic
│   └── reporting.py          # Plotting and performance reporting
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_data_acquisition.py
│   ├── test_data_processing.py
│   └── test_analysis_engine.py
├── .gitignore                # Specifies intentionally untracked files that Git should ignore
├── main.py                   # CLI entry point
├── README.md                 # This file
└── requirements.txt          # Project dependencies
```

## Setup and Installation

### Prerequisites

*   Python 3.8 or higher.
*   Git (for cloning the repository).

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd akshare_quant_analysis_system
    ```
    (Replace `<repository_url>` with the actual URL of the repository if applicable for users).

2.  **Create and Activate a Virtual Environment:**
    *   It's highly recommended to use a virtual environment to manage project dependencies.

    *   Create a virtual environment (e.g., named `venv`):
        ```bash
        python3 -m venv venv
        ```

    *   Activate the virtual environment:
        *   On macOS and Linux:
            ```bash
            source venv/bin/activate
            ```
        *   On Windows:
            ```bash
            venv\Scripts\activate
            ```

3.  **Install Dependencies:**
    *   Ensure your virtual environment is activated.
    *   Install the required packages from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

## Usage (CLI Examples)

The system is operated via `main.py`. Here are examples for each command:

### 1. Fetch Data

This command fetches historical data for a specified stock or index and saves it to a CSV file.

*   `--type`: Type of data (`stock` or `index`).
*   `--code`: Stock code (e.g., "000001" for Ping An Bank) or Index code (e.g., "sh000001" for SSE Composite Index).
*   `--start_date`: Start date for data (YYYY-MM-DD).
*   `--end_date`: End date for data (YYYY-MM-DD).
*   `--output_file`: Path to save the fetched data (e.g., `data/<code>_data.csv`).

**Example:** Fetch stock data for Ping An Bank (000001).
```bash
python main.py fetch --type stock --code "000001" --start_date "2023-01-01" --end_date "2023-12-31" --output_file "data/000001_stock_data.csv"
```

Fetch index data for SSE Composite Index (sh000001).
```bash
python main.py fetch --type index --code "sh000001" --start_date "2023-01-01" --end_date "2023-12-31" --output_file "data/sh000001_index_data.csv"
```

### 2. Analyze Data

This command processes a CSV data file, calculates technical indicators, generates trading signals, and saves the augmented data.

*   `--input_file`: Path to the input CSV data file (previously fetched).
*   `--output_file`: Path to save the analyzed data with indicators and signals.
*   Strategy arguments (optional):
    *   `--sma_short <window>`: Short window for SMA.
    *   `--sma_long <window>`: Long window for SMA.
    *   `--rsi_window <window>`: Window for RSI.
    *   `--rsi_oversold <value>`: RSI oversold threshold (default: 30).
    *   `--rsi_overbought <value>`: RSI overbought threshold (default: 70).
    *   `--bb_window <window>`: Window for Bollinger Bands.
    *   `--bb_std_dev <value>`: Standard deviations for Bollinger Bands (default: 2).
*   `--combine_strategy <strategy>`: How to combine signals if multiple strategies are used (`majority` or `unanimous`, default: `majority`).

**Example:** Analyze the fetched Ping An Bank data using SMA (10-day and 30-day) and RSI (14-day) strategies.
```bash
python main.py analyze --input_file "data/000001_stock_data.csv" --sma_short 10 --sma_long 30 --rsi_window 14 --output_file "data/000001_analyzed_data.csv" --combine_strategy majority
```

### 3. Generate Report

This command generates a performance report and visualizations from the analyzed data.

*   `--input_file`: Path to the analyzed CSV data file (containing signals).
*   `--signal_column <name>`: Name of the signal column to use for the report (default: `combined_signal`). Other options could be `sma_signal`, `rsi_signal`, etc., if they exist in the file.
*   `--price_column <name>`: Name of the price column used for plotting and performance calculation (default: `close`).
*   `--initial_capital <amount>`: Initial capital for performance summary (default: 100000.0).

**Example:** Generate a report for the analyzed Ping An Bank data using the 'combined_signal'.
```bash
python main.py report --input_file "data/000001_analyzed_data.csv" --signal_column combined_signal --initial_capital 100000
```
The report (Markdown file) will be saved in `docs/reports/` and plots (PNG files) in `docs/plots/`.

## Running Tests

To run the unit tests for the system, ensure `pytest` and `pytest-cov` (optional, for coverage) are installed (they are included in `requirements.txt`). Navigate to the project root directory (`akshare_quant_analysis_system/`) and run:

```bash
pytest
```

To include coverage reports:
```bash
pytest --cov=src tests/
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure all tests pass.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License.
(Note: A `LICENSE` file is not currently included in this project.)