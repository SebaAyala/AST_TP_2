"""
Example script demonstrating how to use the TimeSeriesAnalysis class.

This is a minimal example showing the basic usage.
"""

from time_series_analysis import TimeSeriesAnalysis

# Create analysis object with custom series names
tsa = TimeSeriesAnalysis(series_names=['Serie A', 'Serie B', 'Serie C'])

# Generate sample data (or use tsa.load_data('your_file.csv'))
print("Generating sample data...")
data = tsa.generate_sample_data(n_points=150)

# Run complete analysis
print("\nRunning complete time series analysis...\n")
tsa.run_complete_analysis()

print("\nAnalysis completed! Check the generated plots.")
