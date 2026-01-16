"""
Unit tests for Portfolio Simulator and Performance Analyzer.

Run with: pytest tests/test_analysis/test_portfolio_simulator.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.analysis.portfolio_simulator import PortfolioSimulator, quick_simulate
from src.analysis.performance_analyzer import PerformanceAnalyzer, quick_analysis


@pytest.fixture
def simulator():
    """Create a PortfolioSimulator instance."""
    return PortfolioSimulator()


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range('2020-01-01', '2025-01-01', freq='D')
    np.random.seed(42)
    
    returns = np.random.normal(0.0005, 0.01, len(dates))
    prices = 100 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Close': prices,
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)


class TestPortfolioSimulator:
    """Test Portfolio Simulator functions."""
    
    def test_monthly_investment_basic(self, simulator):
        """Test basic monthly investment simulation."""
        result = simulator.simulate_monthly_investment(
            monthly_amount=100,
            years=10,
            include_volatility=False
        )
        
        assert result['total_invested'] == 12000  # 100 * 12 * 10
        assert result['final_value'] > result['total_invested']
        assert result['gains'] > 0
        assert result['percent_gain'] > 0
        assert result['months'] == 120
    
    def test_monthly_investment_zero_amount(self, simulator):
        """Test that zero investment raises error."""
        with pytest.raises(ValueError):
            simulator.simulate_monthly_investment(0, 10)
    
    def test_monthly_investment_negative_years(self, simulator):
        """Test that negative years raises error."""
        with pytest.raises(ValueError):
            simulator.simulate_monthly_investment(100, -5)
    
    def test_monthly_investment_with_volatility(self, simulator):
        """Test simulation with market volatility."""
        result = simulator.simulate_monthly_investment(
            monthly_amount=100,
            years=10,
            include_volatility=True
        )
        
        assert result['final_value'] > 0
        assert len(result['monthly_values']) == 120
    
    def test_lump_sum_basic(self, simulator):
        """Test lump sum investment."""
        result = simulator.simulate_lump_sum(
            initial_amount=10000,
            years=10,
            include_volatility=False
        )
        
        assert result['total_invested'] == 10000
        assert result['final_value'] > 10000
        assert result['gains'] > 0
    
    def test_lump_sum_zero_amount(self, simulator):
        """Test that zero lump sum raises error."""
        with pytest.raises(ValueError):
            simulator.simulate_lump_sum(0, 10)
    
    def test_monte_carlo_basic(self, simulator):
        """Test Monte Carlo simulation."""
        result = simulator.monte_carlo_simulation(
            monthly_amount=100,
            years=5,
            num_simulations=100
        )
        
        assert result['num_simulations'] == 100
        assert result['median_outcome'] > 0
        assert result['best_case'] > result['median_outcome']
        assert result['worst_case'] < result['median_outcome']
        assert len(result['all_outcomes']) == 100
    
    def test_monte_carlo_percentiles(self, simulator):
        """Test that Monte Carlo percentiles are ordered correctly."""
        result = simulator.monte_carlo_simulation(
            monthly_amount=100,
            years=5,
            num_simulations=1000
        )
        
        assert result['worst_case'] < result['percentile_25']
        assert result['percentile_25'] < result['median_outcome']
        assert result['median_outcome'] < result['percentile_75']
        assert result['percentile_75'] < result['best_case']
    
    def test_compare_strategies(self, simulator):
        """Test strategy comparison."""
        result = simulator.compare_strategies(
            monthly_amount=200,
            lump_sum_amount=24000,
            years=10
        )
        
        assert 'dca' in result
        assert 'lump_sum' in result
        assert 'winner' in result
        assert result['winner'] in ['dca', 'lump_sum']
        assert result['difference'] >= 0
    
    def test_retirement_needs_basic(self, simulator):
        """Test retirement needs calculation."""
        result = simulator.calculate_retirement_needs(
            current_age=25,
            retirement_age=65,
            desired_annual_income=50000,
            current_savings=0
        )
        
        assert result['years_to_retirement'] == 40
        assert result['target_portfolio'] > 0
        assert result['monthly_savings_needed'] > 0
    
    def test_retirement_needs_with_savings(self, simulator):
        """Test retirement calculation with existing savings."""
        result = simulator.calculate_retirement_needs(
            current_age=25,
            retirement_age=65,
            desired_annual_income=50000,
            current_savings=100000
        )
        
        # With savings, should need less monthly
        assert result['current_savings'] == 100000
        assert result['current_savings_at_retirement'] > 100000
    
    def test_retirement_invalid_age(self, simulator):
        """Test that invalid retirement age raises error."""
        with pytest.raises(ValueError):
            simulator.calculate_retirement_needs(
                current_age=65,
                retirement_age=60,
                desired_annual_income=50000
            )
    
    def test_historical_backtest(self, simulator, sample_stock_data):
        """Test historical backtesting."""
        result = simulator.historical_backtest(
            monthly_amount=100,
            start_date='2020-01-01',
            end_date='2024-12-31',
            stock_data=sample_stock_data
        )
        
        assert result['total_invested'] > 0
        assert result['final_value'] > 0
        assert result['months'] > 0
        assert 'portfolio_history' in result
    
    def test_quick_simulate_function(self):
        """Test convenience function."""
        result = quick_simulate(100, 10)
        
        assert result['total_invested'] == 12000
        assert result['final_value'] > 0


class TestPerformanceAnalyzer:
    """Test Performance Analyzer functions."""
    
    def test_initialization(self, sample_stock_data):
        """Test analyzer initialization."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        
        assert analyzer.data is not None
        assert analyzer.returns is not None
        assert len(analyzer.returns) > 0
    
    def test_initialization_no_close_column(self):
        """Test that missing Close column raises error."""
        bad_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103]
        })
        
        with pytest.raises(ValueError):
            PerformanceAnalyzer(bad_data)
    
    def test_total_return(self, sample_stock_data):
        """Test total return calculation."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        total_return = analyzer.get_total_return()
        
        assert isinstance(total_return, float)
        # With random seed, should be positive
        assert total_return != 0
    
    def test_annualized_return(self, sample_stock_data):
        """Test annualized return calculation."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        annual_return = analyzer.get_annualized_return()
        
        assert isinstance(annual_return, float)
    
    def test_volatility(self, sample_stock_data):
        """Test volatility calculation."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        
        annual_vol = analyzer.get_volatility(annualized=True)
        daily_vol = analyzer.get_volatility(annualized=False)
        
        assert annual_vol > daily_vol
        assert annual_vol > 0
    
    def test_sharpe_ratio(self, sample_stock_data):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        sharpe = analyzer.get_sharpe_ratio()
        
        assert isinstance(sharpe, float)
        # Sharpe can be negative if returns < risk-free rate
    
    def test_max_drawdown(self, sample_stock_data):
        """Test maximum drawdown calculation."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        dd = analyzer.get_max_drawdown()
        
        assert 'max_drawdown' in dd
        assert 'peak_date' in dd
        assert 'trough_date' in dd
        assert dd['max_drawdown'] <= 0  # Drawdown is negative
    
    def test_monthly_returns(self, sample_stock_data):
        """Test monthly returns calculation."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        monthly = analyzer.get_monthly_returns()
        
        assert isinstance(monthly, pd.Series)
        assert len(monthly) > 0
    
    def test_best_and_worst_days(self, sample_stock_data):
        """Test best and worst day identification."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        result = analyzer.get_best_and_worst_days(n=5)
        
        assert 'best_days' in result
        assert 'worst_days' in result
        assert len(result['best_days']) == 5
        assert len(result['worst_days']) == 5
    
    def test_win_rate(self, sample_stock_data):
        """Test win rate calculation."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        wr = analyzer.get_win_rate()
        
        assert 'win_rate' in wr
        assert 'positive_days' in wr
        assert 'negative_days' in wr
        assert 0 <= wr['win_rate'] <= 100
    
    def test_calculate_all_metrics(self, sample_stock_data):
        """Test calculating all metrics at once."""
        analyzer = PerformanceAnalyzer(sample_stock_data)
        metrics = analyzer.calculate_all_metrics()
        
        required_keys = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'win_rate',
            'current_price', 'start_price'
        ]
        
        for key in required_keys:
            assert key in metrics
    
    def test_compare_stocks(self, sample_stock_data):
        """Test stock comparison."""
        # Create multiple "stocks" with same data
        data_dict = {
            'STOCK1': sample_stock_data.copy(),
            'STOCK2': sample_stock_data.copy() * 1.1,  # Slightly different
            'STOCK3': sample_stock_data.copy() * 0.9
        }
        
        comparison = PerformanceAnalyzer.compare_stocks(data_dict)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert 'Symbol' in comparison.columns
        assert 'Total Return %' in comparison.columns
    
    def test_quick_analysis_function(self, sample_stock_data):
        """Test convenience function."""
        metrics = quick_analysis(sample_stock_data)
        
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics


class TestIntegration:
    """Integration tests combining simulator and analyzer."""
    
    def test_simulate_and_analyze(self, simulator, sample_stock_data):
        """Test simulating an investment and analyzing results."""
        # Run backtest
        backtest = simulator.historical_backtest(
            monthly_amount=100,
            start_date='2020-01-01',
            end_date='2024-12-31',
            stock_data=sample_stock_data
        )
        
        # Analyze the stock used
        analyzer = PerformanceAnalyzer(sample_stock_data)
        metrics = analyzer.calculate_all_metrics()
        
        # Both should have positive results (given our sample data)
        assert backtest['final_value'] > 0
        assert metrics['annualized_return'] != 0
    
    def test_monte_carlo_variance(self, simulator):
        """Test that Monte Carlo produces reasonable variance."""
        result = simulator.monte_carlo_simulation(
            monthly_amount=100,
            years=10,
            num_simulations=1000
        )
        
        # Range should be significant
        range_width = result['best_case'] - result['worst_case']
        median = result['median_outcome']
        
        # Range should be at least 20% of median
        assert range_width > median * 0.2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_investment(self, simulator):
        """Test with very small investment amount."""
        result = simulator.simulate_monthly_investment(1, 1)
        assert result['total_invested'] == 12
        assert result['final_value'] > 0
    
    def test_very_long_horizon(self, simulator):
        """Test with very long time horizon."""
        result = simulator.simulate_monthly_investment(100, 40)
        assert result['months'] == 480
        assert result['final_value'] > result['total_invested']
    
    def test_negative_returns(self, simulator):
        """Test simulation with negative expected returns."""
        result = simulator.simulate_monthly_investment(
            monthly_amount=100,
            years=5,
            annual_return=-0.05,  # -5% per year
            include_volatility=False
        )
        
        # Should still complete successfully
        assert result['final_value'] > 0
        # But might be less than invested
    
    def test_zero_volatility(self, simulator):
        """Test simulation with zero volatility."""
        result = simulator.simulate_monthly_investment(
            monthly_amount=100,
            years=10,
            annual_volatility=0,
            include_volatility=True
        )
        
        assert result['final_value'] > 0
    
    def test_single_day_data(self):
        """Test analyzer with single day of data."""
        single_day = pd.DataFrame({
            'Close': [100],
            'Open': [99],
            'High': [101],
            'Low': [98]
        }, index=[datetime.now()])
        
        analyzer = PerformanceAnalyzer(single_day)
        # Should not crash, but returns will be zero/empty
        assert len(analyzer.returns) == 0


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
