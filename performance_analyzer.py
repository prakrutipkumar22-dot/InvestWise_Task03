"""
Performance Analyzer Module
Analyzes stock and portfolio performance with various metrics.

Author: InvestWise Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes investment performance and calculates key metrics.
    
    Features:
    - Returns calculation (daily, monthly, annual)
    - Volatility and risk metrics
    - Sharpe ratio
    - Maximum drawdown
    - Correlation analysis
    - Performance comparisons
    
    Example:
        >>> analyzer = PerformanceAnalyzer(stock_data)
        >>> metrics = analyzer.calculate_all_metrics()
        >>> print(f"Annual return: {metrics['annual_return']:.2f}%")
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with stock data.
        
        Args:
            data: DataFrame with at least 'Close' column and DatetimeIndex
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must have 'Close' column")
        
        self.data = data.copy()
        self.returns = self._calculate_returns()
        logger.info(f"PerformanceAnalyzer initialized with {len(data)} records")
    
    def _calculate_returns(self) -> pd.Series:
        """Calculate daily returns."""
        return self.data['Close'].pct_change().dropna()
    
    def get_total_return(self) -> float:
        """
        Calculate total return over the entire period.
        
        Returns:
            Total return as percentage
            
        Example:
            >>> analyzer = PerformanceAnalyzer(data)
            >>> total = analyzer.get_total_return()
            >>> print(f"Total return: {total:.2f}%")
        """
        start_price = self.data['Close'].iloc[0]
        end_price = self.data['Close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        return round(total_return, 2)
    
    def get_annualized_return(self) -> float:
        """
        Calculate annualized return (CAGR).
        
        Returns:
            Annualized return as percentage
        """
        start_price = self.data['Close'].iloc[0]
        end_price = self.data['Close'].iloc[-1]
        
        # Calculate number of years
        days = (self.data.index[-1] - self.data.index[0]).days
        years = days / 365.25
        
        if years == 0:
            return 0.0
        
        # CAGR = (End Value / Start Value) ^ (1 / Years) - 1
        cagr = ((end_price / start_price) ** (1 / years) - 1) * 100
        return round(cagr, 2)
    
    def get_volatility(self, annualized: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            annualized: If True, annualize the volatility
            
        Returns:
            Volatility as percentage
            
        Example:
            >>> analyzer = PerformanceAnalyzer(data)
            >>> vol = analyzer.get_volatility()
            >>> print(f"Annual volatility: {vol:.2f}%")
        """
        daily_vol = self.returns.std()
        
        if annualized:
            # Annualize: multiply by sqrt(252) trading days
            vol = daily_vol * np.sqrt(252) * 100
        else:
            vol = daily_vol * 100
        
        return round(vol, 2)
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Sharpe Ratio = (Return - Risk Free Rate) / Volatility
        Higher is better (more return per unit of risk).
        
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.04 for 4%)
            
        Returns:
            Sharpe ratio
            
        Example:
            >>> analyzer = PerformanceAnalyzer(data)
            >>> sharpe = analyzer.get_sharpe_ratio()
            >>> print(f"Sharpe ratio: {sharpe:.2f}")
            >>> if sharpe > 1:
            ...     print("Good risk-adjusted returns!")
        """
        annual_return = self.get_annualized_return() / 100  # Convert to decimal
        annual_vol = self.get_volatility() / 100  # Convert to decimal
        
        if annual_vol == 0:
            return 0.0
        
        sharpe = (annual_return - risk_free_rate) / annual_vol
        return round(sharpe, 2)
    
    def get_max_drawdown(self) -> Dict:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Returns:
            Dictionary with:
                - max_drawdown: Maximum drawdown percentage
                - peak_date: Date of the peak
                - trough_date: Date of the trough
                - recovery_date: Date when it recovered (or None)
                
        Example:
            >>> analyzer = PerformanceAnalyzer(data)
            >>> dd = analyzer.get_max_drawdown()
            >>> print(f"Max drawdown: {dd['max_drawdown']:.2f}%")
            >>> print(f"From {dd['peak_date']} to {dd['trough_date']}")
        """
        # Calculate cumulative returns
        cumulative = (1 + self.returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max * 100
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find the peak before the trough
        peak_idx = cumulative.loc[:max_dd_idx].idxmax()
        
        # Try to find recovery date (when it exceeded the previous peak)
        recovery_idx = None
        after_trough = cumulative.loc[max_dd_idx:]
        peak_value = cumulative.loc[peak_idx]
        recovery_mask = after_trough >= peak_value
        
        if recovery_mask.any():
            recovery_idx = after_trough[recovery_mask].index[0]
        
        return {
            'max_drawdown': round(max_dd, 2),
            'peak_date': str(peak_idx.date()),
            'trough_date': str(max_dd_idx.date()),
            'recovery_date': str(recovery_idx.date()) if recovery_idx else None,
            'days_to_recover': (recovery_idx - max_dd_idx).days if recovery_idx else None
        }
    
    def get_monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns.
        
        Returns:
            Series of monthly returns
        """
        monthly_close = self.data['Close'].resample('M').last()
        monthly_returns = monthly_close.pct_change().dropna() * 100
        return monthly_returns
    
    def get_best_and_worst_days(self, n: int = 5) -> Dict:
        """
        Find best and worst performing days.
        
        Args:
            n: Number of days to return
            
        Returns:
            Dictionary with best and worst days
        """
        returns_pct = self.returns * 100
        
        best_days = returns_pct.nlargest(n)
        worst_days = returns_pct.nsmallest(n)
        
        return {
            'best_days': [
                {'date': str(idx.date()), 'return': round(val, 2)}
                for idx, val in best_days.items()
            ],
            'worst_days': [
                {'date': str(idx.date()), 'return': round(val, 2)}
                for idx, val in worst_days.items()
            ]
        }
    
    def get_win_rate(self) -> Dict:
        """
        Calculate win rate (percentage of positive days).
        
        Returns:
            Dictionary with win rate statistics
        """
        positive_days = (self.returns > 0).sum()
        total_days = len(self.returns)
        win_rate = (positive_days / total_days) * 100
        
        avg_gain = self.returns[self.returns > 0].mean() * 100
        avg_loss = self.returns[self.returns < 0].mean() * 100
        
        return {
            'win_rate': round(win_rate, 2),
            'positive_days': int(positive_days),
            'negative_days': int(total_days - positive_days),
            'total_days': int(total_days),
            'avg_gain': round(avg_gain, 2),
            'avg_loss': round(avg_loss, 2),
            'gain_loss_ratio': round(abs(avg_gain / avg_loss), 2) if avg_loss != 0 else 0
        }
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all performance metrics at once.
        
        Returns:
            Dictionary with all metrics
            
        Example:
            >>> analyzer = PerformanceAnalyzer(data)
            >>> metrics = analyzer.calculate_all_metrics()
            >>> for key, value in metrics.items():
            ...     print(f"{key}: {value}")
        """
        return {
            'total_return': self.get_total_return(),
            'annualized_return': self.get_annualized_return(),
            'volatility': self.get_volatility(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'max_drawdown': self.get_max_drawdown(),
            'win_rate': self.get_win_rate(),
            'current_price': round(self.data['Close'].iloc[-1], 2),
            'start_price': round(self.data['Close'].iloc[0], 2),
            'high': round(self.data['Close'].max(), 2),
            'low': round(self.data['Close'].min(), 2),
            'data_points': len(self.data),
            'start_date': str(self.data.index[0].date()),
            'end_date': str(self.data.index[-1].date())
        }
    
    @staticmethod
    def compare_stocks(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare performance of multiple stocks.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            DataFrame with comparison metrics
            
        Example:
            >>> from src.data.stock_fetcher import StockFetcher
            >>> fetcher = StockFetcher()
            >>> data = {
            ...     'AAPL': fetcher.get_historical_data('AAPL', '1y'),
            ...     'MSFT': fetcher.get_historical_data('MSFT', '1y'),
            ...     'GOOGL': fetcher.get_historical_data('GOOGL', '1y')
            ... }
            >>> comparison = PerformanceAnalyzer.compare_stocks(data)
            >>> print(comparison)
        """
        results = []
        
        for symbol, data in data_dict.items():
            analyzer = PerformanceAnalyzer(data)
            metrics = analyzer.calculate_all_metrics()
            
            results.append({
                'Symbol': symbol,
                'Total Return %': metrics['total_return'],
                'Annual Return %': metrics['annualized_return'],
                'Volatility %': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown %': metrics['max_drawdown']['max_drawdown'],
                'Win Rate %': metrics['win_rate']['win_rate'],
                'Current Price': metrics['current_price']
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('Annual Return %', ascending=False)
        return df


def quick_analysis(data: pd.DataFrame) -> Dict:
    """
    Quick performance analysis.
    
    Example:
        >>> from src.data.stock_fetcher import get_stock_data
        >>> data = get_stock_data('AAPL', '1y')
        >>> metrics = quick_analysis(data)
        >>> print(f"Return: {metrics['annualized_return']:.2f}%")
    """
    analyzer = PerformanceAnalyzer(data)
    return analyzer.calculate_all_metrics()


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Performance Analyzer Demo")
    print("="*60)
    
    # Create sample data
    print("\nCreating sample stock data...")
    dates = pd.date_range('2020-01-01', '2025-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate stock price with trend and volatility
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * (1 + returns).cumprod()
    
    sample_data = pd.DataFrame({
        'Close': prices,
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Analyze
    analyzer = PerformanceAnalyzer(sample_data)
    
    print("\n1. Basic Returns")
    print(f"   Total Return: {analyzer.get_total_return():.2f}%")
    print(f"   Annualized Return: {analyzer.get_annualized_return():.2f}%")
    
    print("\n2. Risk Metrics")
    print(f"   Volatility: {analyzer.get_volatility():.2f}%")
    print(f"   Sharpe Ratio: {analyzer.get_sharpe_ratio():.2f}")
    
    print("\n3. Drawdown Analysis")
    dd = analyzer.get_max_drawdown()
    print(f"   Max Drawdown: {dd['max_drawdown']:.2f}%")
    print(f"   Peak Date: {dd['peak_date']}")
    print(f"   Trough Date: {dd['trough_date']}")
    
    print("\n4. Win Rate")
    wr = analyzer.get_win_rate()
    print(f"   Win Rate: {wr['win_rate']:.2f}%")
    print(f"   Avg Gain: {wr['avg_gain']:.2f}%")
    print(f"   Avg Loss: {wr['avg_loss']:.2f}%")
    
    print("\n5. All Metrics")
    metrics = analyzer.calculate_all_metrics()
    for key, value in metrics.items():
        if key not in ['max_drawdown', 'win_rate']:
            print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("Demo complete!")
