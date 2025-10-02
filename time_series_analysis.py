"""
Time Series Analysis - TP 2
Análisis de Series Temporales 2025

This script performs time series analysis on three different series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class TimeSeriesAnalysis:
    """Class for performing time series analysis on multiple series."""
    
    def __init__(self, series_names=None):
        """
        Initialize the TimeSeriesAnalysis object.
        
        Parameters:
        -----------
        series_names : list, optional
            Names for the time series (default: ['Series 1', 'Series 2', 'Series 3'])
        """
        self.series_names = series_names or ['Series 1', 'Series 2', 'Series 3']
        self.series = {}
        
    def generate_sample_data(self, n_points=200):
        """
        Generate sample time series data for demonstration.
        
        Parameters:
        -----------
        n_points : int
            Number of data points for each series
            
        Returns:
        --------
        dict : Dictionary containing the three time series
        """
        np.random.seed(42)
        t = np.arange(n_points)
        
        # Series 1: Trend + Seasonality + Noise
        trend1 = 0.5 * t
        seasonality1 = 10 * np.sin(2 * np.pi * t / 12)
        noise1 = np.random.normal(0, 5, n_points)
        series1 = trend1 + seasonality1 + noise1
        
        # Series 2: Stationary with AR(1) component
        series2 = np.zeros(n_points)
        series2[0] = np.random.normal(0, 1)
        for i in range(1, n_points):
            series2[i] = 0.7 * series2[i-1] + np.random.normal(0, 3)
        
        # Series 3: Random walk (non-stationary)
        series3 = np.cumsum(np.random.normal(0, 2, n_points))
        
        # Store in DataFrame
        self.series = pd.DataFrame({
            self.series_names[0]: series1,
            self.series_names[1]: series2,
            self.series_names[2]: series3
        })
        
        return self.series
    
    def load_data(self, filepath, column_names=None):
        """
        Load time series data from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        column_names : list, optional
            Names of columns to use as series
        """
        df = pd.read_csv(filepath)
        if column_names:
            self.series = df[column_names]
            self.series_names = column_names
        else:
            self.series = df
            self.series_names = list(df.columns)
        return self.series
    
    def descriptive_statistics(self):
        """
        Calculate descriptive statistics for all series.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with descriptive statistics
        """
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        
        stats_df = self.series.describe()
        print(stats_df)
        
        # Additional statistics
        print("\nAdditional Statistics:")
        print(f"{'Statistic':<20} {self.series_names[0]:<15} {self.series_names[1]:<15} {self.series_names[2]:<15}")
        print("-" * 70)
        
        for name in self.series_names:
            series = self.series[name]
            skewness = stats.skew(series)
            kurtosis_val = stats.kurtosis(series)
            
            print(f"{'Skewness':<20} {skewness:.4f}")
            print(f"{'Kurtosis':<20} {kurtosis_val:.4f}")
            print("-" * 70)
        
        return stats_df
    
    def plot_series(self, save_path=None):
        """
        Plot all time series.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for idx, name in enumerate(self.series_names):
            axes[idx].plot(self.series[name], linewidth=1.5)
            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        print("\nTime series plots generated.")
    
    def test_stationarity(self):
        """
        Perform Augmented Dickey-Fuller test for stationarity on all series.
        
        Returns:
        --------
        dict : Dictionary with test results for each series
        """
        print("\n" + "="*60)
        print("STATIONARITY TESTS (Augmented Dickey-Fuller)")
        print("="*60)
        
        results = {}
        
        for name in self.series_names:
            series = self.series[name].dropna()
            
            # Perform ADF test
            adf_result = adfuller(series, autolag='AIC')
            
            results[name] = {
                'ADF Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Lags Used': adf_result[2],
                'Observations': adf_result[3],
                'Critical Values': adf_result[4]
            }
            
            print(f"\n{name}:")
            print(f"  ADF Statistic: {adf_result[0]:.6f}")
            print(f"  p-value: {adf_result[1]:.6f}")
            print(f"  Lags Used: {adf_result[2]}")
            print(f"  Number of Observations: {adf_result[3]}")
            print(f"  Critical Values:")
            for key, value in adf_result[4].items():
                print(f"    {key}: {value:.3f}")
            
            # Interpret result
            if adf_result[1] <= 0.05:
                print(f"  => Series is STATIONARY (reject H0 at 5% significance)")
            else:
                print(f"  => Series is NON-STATIONARY (fail to reject H0 at 5% significance)")
        
        return results
    
    def plot_acf_pacf(self, lags=40, save_path=None):
        """
        Plot ACF and PACF for all series.
        
        Parameters:
        -----------
        lags : int
            Number of lags to include
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        for idx, name in enumerate(self.series_names):
            series = self.series[name].dropna()
            
            # ACF plot
            plot_acf(series, lags=lags, ax=axes[idx, 0])
            axes[idx, 0].set_title(f'{name} - ACF')
            
            # PACF plot
            plot_pacf(series, lags=lags, ax=axes[idx, 1])
            axes[idx, 1].set_title(f'{name} - PACF')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        print("\nACF and PACF plots generated.")
    
    def plot_distributions(self, save_path=None):
        """
        Plot histograms and Q-Q plots for all series.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        for idx, name in enumerate(self.series_names):
            series = self.series[name].dropna()
            
            # Histogram
            axes[idx, 0].hist(series, bins=30, edgecolor='black', alpha=0.7)
            axes[idx, 0].set_title(f'{name} - Histogram')
            axes[idx, 0].set_xlabel('Value')
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(series, dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'{name} - Q-Q Plot')
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        print("\nDistribution plots generated.")
    
    def run_complete_analysis(self):
        """
        Run a complete time series analysis on all series.
        """
        print("\n" + "="*60)
        print("TIME SERIES ANALYSIS - TP 2")
        print("Análisis de Series Temporales 2025")
        print("="*60)
        
        # Descriptive statistics
        self.descriptive_statistics()
        
        # Plot series
        self.plot_series()
        
        # Stationarity tests
        self.test_stationarity()
        
        # ACF and PACF plots
        self.plot_acf_pacf()
        
        # Distribution plots
        self.plot_distributions()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)


def main():
    """Main function to run the time series analysis."""
    # Create analysis object
    tsa = TimeSeriesAnalysis()
    
    # Generate sample data (or load from file)
    print("Generating sample time series data...")
    tsa.generate_sample_data(n_points=200)
    
    # Run complete analysis
    tsa.run_complete_analysis()
    
    # Save the data to CSV for reference
    tsa.series.to_csv('time_series_data.csv', index=False)
    print("\nData saved to 'time_series_data.csv'")


if __name__ == "__main__":
    main()
