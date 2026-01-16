"""
Portfolio Simulator Module
Simulates investment growth and portfolio performance.

Author: InvestWise Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """
    Simulates investment portfolio growth and performance.
    
    Features:
    - Dollar-cost averaging simulation
    - Lump sum investment simulation
    - Monte Carlo simulation for future projections
    - Historical backtesting with real data
    - Multiple investment strategies
    - Risk-adjusted returns
    
    Example:
        >>> sim = PortfolioSimulator()
        >>> result = sim.simulate_monthly_investment(
        ...     monthly_amount=200,
        ...     years=10,
        ...     annual_return=0.10
        ... )
        >>> print(f"Final value: ${result['final_value']:,.2f}")
    """
    
    def __init__(self):
        """Initialize the portfolio simulator."""
        logger.info("PortfolioSimulator initialized")
    
    def simulate_monthly_investment(
        self,
        monthly_amount: float,
        years: int,
        annual_return: float = 0.10,
        annual_volatility: float = 0.15,
        include_volatility: bool = True
    ) -> Dict:
        """
        Simulate monthly investment with compound returns.
        
        This is the classic "dollar-cost averaging" strategy where you
        invest a fixed amount each month.
        
        Args:
            monthly_amount: Amount to invest each month ($)
            years: Investment time horizon
            annual_return: Expected annual return (e.g., 0.10 for 10%)
            annual_volatility: Annual volatility/std deviation (e.g., 0.15 for 15%)
            include_volatility: Whether to include market volatility in simulation
            
        Returns:
            Dictionary containing:
                - total_invested: Total amount invested
                - final_value: Final portfolio value
                - gains: Total gains (final_value - total_invested)
                - percent_gain: Percentage gain
                - annualized_return: Annualized return rate
                - monthly_values: List of portfolio values each month
                - monthly_contributions: List of contributions each month
                
        Example:
            >>> sim = PortfolioSimulator()
            >>> result = sim.simulate_monthly_investment(100, 10)
            >>> print(f"Invested ${result['total_invested']:,.2f}")
            >>> print(f"Grew to ${result['final_value']:,.2f}")
        """
        # Validate inputs
        if monthly_amount <= 0:
            raise ValueError("Monthly amount must be positive")
        if years <= 0:
            raise ValueError("Years must be positive")
        if annual_return < -1 or annual_return > 5:
            raise ValueError("Annual return seems unrealistic")
        
        months = years * 12
        monthly_return = annual_return / 12
        monthly_volatility = annual_volatility / np.sqrt(12) if include_volatility else 0
        
        # Simulate each month
        portfolio_values = []
        contributions = []
        current_value = 0
        total_invested = 0
        
        for month in range(months):
            # Add monthly contribution
            current_value += monthly_amount
            total_invested += monthly_amount
            contributions.append(monthly_amount)
            
            # Apply return (with optional volatility)
            if include_volatility:
                # Add random noise based on volatility
                random_return = np.random.normal(monthly_return, monthly_volatility)
                current_value *= (1 + random_return)
            else:
                # Deterministic return
                current_value *= (1 + monthly_return)
            
            portfolio_values.append(current_value)
        
        final_value = portfolio_values[-1]
        gains = final_value - total_invested
        percent_gain = (gains / total_invested) * 100
        
        # Calculate annualized return
        # Formula: (final_value / total_invested) ^ (1/years) - 1
        # But for DCA, we use IRR approximation
        annualized_return = ((final_value / total_invested) ** (1/years) - 1) * 100
        
        return {
            'total_invested': round(total_invested, 2),
            'final_value': round(final_value, 2),
            'gains': round(gains, 2),
            'percent_gain': round(percent_gain, 2),
            'annualized_return': round(annualized_return, 2),
            'monthly_values': [round(v, 2) for v in portfolio_values],
            'monthly_contributions': contributions,
            'months': months,
            'strategy': 'dollar_cost_averaging'
        }
    
    def simulate_lump_sum(
        self,
        initial_amount: float,
        years: int,
        annual_return: float = 0.10,
        annual_volatility: float = 0.15,
        include_volatility: bool = True
    ) -> Dict:
        """
        Simulate lump sum investment (invest all at once).
        
        Args:
            initial_amount: One-time investment amount
            years: Investment time horizon
            annual_return: Expected annual return
            annual_volatility: Annual volatility
            include_volatility: Whether to include market volatility
            
        Returns:
            Dictionary with simulation results
            
        Example:
            >>> sim = PortfolioSimulator()
            >>> result = sim.simulate_lump_sum(10000, 10, 0.10)
            >>> print(f"$10,000 grew to ${result['final_value']:,.2f}")
        """
        if initial_amount <= 0:
            raise ValueError("Initial amount must be positive")
        if years <= 0:
            raise ValueError("Years must be positive")
        
        months = years * 12
        monthly_return = annual_return / 12
        monthly_volatility = annual_volatility / np.sqrt(12) if include_volatility else 0
        
        portfolio_values = []
        current_value = initial_amount
        
        for month in range(months):
            if include_volatility:
                random_return = np.random.normal(monthly_return, monthly_volatility)
                current_value *= (1 + random_return)
            else:
                current_value *= (1 + monthly_return)
            
            portfolio_values.append(current_value)
        
        final_value = portfolio_values[-1]
        gains = final_value - initial_amount
        percent_gain = (gains / initial_amount) * 100
        annualized_return = ((final_value / initial_amount) ** (1/years) - 1) * 100
        
        return {
            'total_invested': round(initial_amount, 2),
            'final_value': round(final_value, 2),
            'gains': round(gains, 2),
            'percent_gain': round(percent_gain, 2),
            'annualized_return': round(annualized_return, 2),
            'monthly_values': [round(v, 2) for v in portfolio_values],
            'months': months,
            'strategy': 'lump_sum'
        }
    
    def monte_carlo_simulation(
        self,
        monthly_amount: float,
        years: int,
        annual_return: float = 0.10,
        annual_volatility: float = 0.15,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Run Monte Carlo simulation to show range of possible outcomes.
        
        This runs many simulations with random market movements to show
        the distribution of possible results.
        
        Args:
            monthly_amount: Monthly investment amount
            years: Investment time horizon
            annual_return: Expected annual return
            annual_volatility: Market volatility
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary containing:
                - median_outcome: 50th percentile result
                - best_case: 95th percentile (optimistic)
                - worst_case: 5th percentile (pessimistic)
                - likely_range: 25th to 75th percentile
                - all_outcomes: All simulation results
                
        Example:
            >>> sim = PortfolioSimulator()
            >>> result = sim.monte_carlo_simulation(200, 10, num_simulations=1000)
            >>> print(f"Median outcome: ${result['median_outcome']:,.2f}")
            >>> print(f"Best case (95%): ${result['best_case']:,.2f}")
            >>> print(f"Worst case (5%): ${result['worst_case']:,.2f}")
        """
        logger.info(f"Running Monte Carlo with {num_simulations} simulations...")
        
        final_values = []
        
        for _ in range(num_simulations):
            result = self.simulate_monthly_investment(
                monthly_amount=monthly_amount,
                years=years,
                annual_return=annual_return,
                annual_volatility=annual_volatility,
                include_volatility=True
            )
            final_values.append(result['final_value'])
        
        final_values = np.array(final_values)
        
        return {
            'median_outcome': round(np.percentile(final_values, 50), 2),
            'mean_outcome': round(np.mean(final_values), 2),
            'best_case': round(np.percentile(final_values, 95), 2),
            'worst_case': round(np.percentile(final_values, 5), 2),
            'percentile_25': round(np.percentile(final_values, 25), 2),
            'percentile_75': round(np.percentile(final_values, 75), 2),
            'std_deviation': round(np.std(final_values), 2),
            'all_outcomes': sorted([round(v, 2) for v in final_values]),
            'num_simulations': num_simulations,
            'total_invested': monthly_amount * years * 12
        }
    
    def compare_strategies(
        self,
        monthly_amount: float,
        lump_sum_amount: float,
        years: int,
        annual_return: float = 0.10,
        annual_volatility: float = 0.15
    ) -> Dict:
        """
        Compare dollar-cost averaging vs. lump sum investment.
        
        Args:
            monthly_amount: Monthly investment for DCA
            lump_sum_amount: One-time investment for lump sum
            years: Investment time horizon
            annual_return: Expected annual return
            annual_volatility: Market volatility
            
        Returns:
            Dictionary comparing both strategies
            
        Example:
            >>> sim = PortfolioSimulator()
            >>> result = sim.compare_strategies(
            ...     monthly_amount=200,
            ...     lump_sum_amount=24000,  # Same total
            ...     years=10
            ... )
            >>> print(f"DCA final: ${result['dca']['final_value']:,.2f}")
            >>> print(f"Lump sum final: ${result['lump_sum']['final_value']:,.2f}")
        """
        # Run both simulations
        dca_result = self.simulate_monthly_investment(
            monthly_amount=monthly_amount,
            years=years,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            include_volatility=False  # Use deterministic for fair comparison
        )
        
        lump_sum_result = self.simulate_lump_sum(
            initial_amount=lump_sum_amount,
            years=years,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            include_volatility=False
        )
        
        # Determine winner
        dca_value = dca_result['final_value']
        lump_value = lump_sum_result['final_value']
        
        if lump_value > dca_value:
            winner = 'lump_sum'
            difference = lump_value - dca_value
        else:
            winner = 'dca'
            difference = dca_value - lump_value
        
        return {
            'dca': dca_result,
            'lump_sum': lump_sum_result,
            'winner': winner,
            'difference': round(difference, 2),
            'percentage_difference': round((difference / min(dca_value, lump_value)) * 100, 2)
        }
    
    def calculate_retirement_needs(
        self,
        current_age: int,
        retirement_age: int,
        desired_annual_income: float,
        current_savings: float = 0,
        annual_return: float = 0.10
    ) -> Dict:
        """
        Calculate how much you need to save for retirement.
        
        Uses the 4% rule: you can safely withdraw 4% of your portfolio annually.
        
        Args:
            current_age: Your current age
            retirement_age: When you want to retire
            desired_annual_income: How much you want per year in retirement
            current_savings: What you've saved already
            annual_return: Expected investment return
            
        Returns:
            Dictionary with retirement planning info
            
        Example:
            >>> sim = PortfolioSimulator()
            >>> result = sim.calculate_retirement_needs(
            ...     current_age=25,
            ...     retirement_age=65,
            ...     desired_annual_income=50000
            ... )
            >>> print(f"Need to save ${result['monthly_savings_needed']:,.2f}/month")
        """
        # Calculate target portfolio (4% rule)
        target_portfolio = desired_annual_income / 0.04
        
        # How much do we already have (with growth)?
        years_to_retirement = retirement_age - current_age
        
        if years_to_retirement <= 0:
            raise ValueError("Retirement age must be greater than current age")
        
        # Grow current savings
        current_savings_at_retirement = current_savings * ((1 + annual_return) ** years_to_retirement)
        
        # How much more do we need?
        additional_needed = max(0, target_portfolio - current_savings_at_retirement)
        
        # Calculate required monthly savings
        if additional_needed > 0:
            # Future value of annuity formula
            # FV = PMT × (((1 + r)^n - 1) / r)
            # Solve for PMT: PMT = FV / (((1 + r)^n - 1) / r)
            r = annual_return / 12  # Monthly rate
            n = years_to_retirement * 12  # Total months
            
            monthly_savings_needed = additional_needed / (((1 + r) ** n - 1) / r)
        else:
            monthly_savings_needed = 0
        
        return {
            'target_portfolio': round(target_portfolio, 2),
            'current_savings': round(current_savings, 2),
            'current_savings_at_retirement': round(current_savings_at_retirement, 2),
            'additional_needed': round(additional_needed, 2),
            'monthly_savings_needed': round(monthly_savings_needed, 2),
            'years_to_retirement': years_to_retirement,
            'desired_annual_income': desired_annual_income,
            'message': self._get_retirement_message(monthly_savings_needed, additional_needed)
        }
    
    def _get_retirement_message(self, monthly_needed: float, additional_needed: float) -> str:
        """Generate friendly retirement planning message."""
        if additional_needed <= 0:
            return "Great news! You're already on track for retirement! 🎉"
        elif monthly_needed < 100:
            return f"Good news! You only need to save ${monthly_needed:.2f}/month to hit your goal."
        elif monthly_needed < 500:
            return f"You'll need to save ${monthly_needed:.2f}/month. That's doable with some budgeting!"
        elif monthly_needed < 1000:
            return f"You'll need to save ${monthly_needed:.2f}/month. Consider increasing income or adjusting retirement plans."
        else:
            return f"Saving ${monthly_needed:.2f}/month is challenging. Consider: working longer, reducing retirement spending, or increasing income."
    
    def historical_backtest(
        self,
        monthly_amount: float,
        start_date: str,
        end_date: str,
        stock_data: pd.DataFrame
    ) -> Dict:
        """
        Backtest a strategy using real historical stock data.
        
        Args:
            monthly_amount: Amount to invest each month
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            stock_data: DataFrame with historical prices (from StockFetcher)
            
        Returns:
            Dictionary with backtest results
            
        Example:
            >>> from src.data.stock_fetcher import StockFetcher
            >>> fetcher = StockFetcher()
            >>> data = fetcher.get_historical_data('SPY', period='10y')
            >>> 
            >>> sim = PortfolioSimulator()
            >>> result = sim.historical_backtest(
            ...     monthly_amount=200,
            ...     start_date='2015-01-01',
            ...     end_date='2025-01-01',
            ...     stock_data=data
            ... )
            >>> print(f"Historical result: ${result['final_value']:,.2f}")
        """
        # Filter data to date range
        mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
        data = stock_data.loc[mask].copy()
        
        if len(data) == 0:
            raise ValueError("No data in specified date range")
        
        # Resample to monthly (last trading day of each month)
        monthly_data = data.resample('M').last()
        
        # Simulate investing
        shares_owned = 0
        total_invested = 0
        portfolio_values = []
        
        for date, row in monthly_data.iterrows():
            price = row['Close']
            
            # Buy shares with monthly investment
            shares_bought = monthly_amount / price
            shares_owned += shares_bought
            total_invested += monthly_amount
            
            # Calculate current portfolio value
            current_value = shares_owned * price
            portfolio_values.append({
                'date': date,
                'value': current_value,
                'invested': total_invested,
                'shares': shares_owned,
                'price': price
            })
        
        final_value = portfolio_values[-1]['value']
        gains = final_value - total_invested
        percent_gain = (gains / total_invested) * 100
        
        years = len(monthly_data) / 12
        annualized_return = ((final_value / total_invested) ** (1/years) - 1) * 100
        
        return {
            'total_invested': round(total_invested, 2),
            'final_value': round(final_value, 2),
            'gains': round(gains, 2),
            'percent_gain': round(percent_gain, 2),
            'annualized_return': round(annualized_return, 2),
            'months': len(monthly_data),
            'years': round(years, 2),
            'shares_owned': round(shares_owned, 4),
            'final_price': round(portfolio_values[-1]['price'], 2),
            'portfolio_history': portfolio_values,
            'start_date': str(monthly_data.index[0].date()),
            'end_date': str(monthly_data.index[-1].date())
        }


# Convenience functions
def quick_simulate(monthly_amount: float, years: int) -> Dict:
    """
    Quick simulation with default parameters.
    
    Example:
        >>> result = quick_simulate(200, 10)
        >>> print(f"${result['final_value']:,.2f}")
    """
    sim = PortfolioSimulator()
    return sim.simulate_monthly_investment(monthly_amount, years)


if __name__ == "__main__":
    # Demo and testing
    print("="*60)
    print("Portfolio Simulator Demo")
    print("="*60)
    
    sim = PortfolioSimulator()
    
    # Test 1: Monthly investment
    print("\n1. Monthly Investment Simulation ($200/month for 10 years)")
    result = sim.simulate_monthly_investment(200, 10, include_volatility=False)
    print(f"   Total Invested: ${result['total_invested']:,.2f}")
    print(f"   Final Value: ${result['final_value']:,.2f}")
    print(f"   Gains: ${result['gains']:,.2f} ({result['percent_gain']:.1f}%)")
    print(f"   Annualized Return: {result['annualized_return']:.2f}%")
    
    # Test 2: Lump sum
    print("\n2. Lump Sum Investment ($24,000 for 10 years)")
    result = sim.simulate_lump_sum(24000, 10, include_volatility=False)
    print(f"   Initial Investment: ${result['total_invested']:,.2f}")
    print(f"   Final Value: ${result['final_value']:,.2f}")
    print(f"   Gains: ${result['gains']:,.2f} ({result['percent_gain']:.1f}%)")
    
    # Test 3: Compare strategies
    print("\n3. Strategy Comparison (DCA vs Lump Sum)")
    result = sim.compare_strategies(200, 24000, 10)
    print(f"   DCA Final: ${result['dca']['final_value']:,.2f}")
    print(f"   Lump Sum Final: ${result['lump_sum']['final_value']:,.2f}")
    print(f"   Winner: {result['winner'].upper()}")
    print(f"   Difference: ${result['difference']:,.2f}")
    
    # Test 4: Monte Carlo
    print("\n4. Monte Carlo Simulation (1000 scenarios)")
    result = sim.monte_carlo_simulation(200, 10, num_simulations=1000)
    print(f"   Median Outcome: ${result['median_outcome']:,.2f}")
    print(f"   Best Case (95%): ${result['best_case']:,.2f}")
    print(f"   Worst Case (5%): ${result['worst_case']:,.2f}")
    print(f"   Likely Range: ${result['percentile_25']:,.2f} - ${result['percentile_75']:,.2f}")
    
    # Test 5: Retirement planning
    print("\n5. Retirement Planning (Age 25 → 65, want $50k/year)")
    result = sim.calculate_retirement_needs(25, 65, 50000, current_savings=5000)
    print(f"   Target Portfolio: ${result['target_portfolio']:,.2f}")
    print(f"   Monthly Savings Needed: ${result['monthly_savings_needed']:,.2f}")
    print(f"   {result['message']}")
    
    print("\n" + "="*60)
    print("Demo complete!")
