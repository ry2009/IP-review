import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Load data with market regimes and first refinement results
df = pd.read_csv('results/enhanced_strategy_results.csv', parse_dates=['date'])
print(f"Data loaded: {df.shape[0]} rows from {df.date.min().strftime('%Y-%m-%d')} to {df.date.max().strftime('%Y-%m-%d')}")

# Create a copy of the original data to compare results
original_df = df.copy()

# ------------------------------------------------------
# 1. Improve Risk Management with Modified Trailing Stops
# ------------------------------------------------------

# Calculate higher frequency volatility for stop calibration
df['volatility_5d'] = df['target'].rolling(window=5).std()
df['volatility_ratio'] = df['volatility_5d'] / df['volatility_10d']
df['volatile_market'] = (df['volatility_ratio'] > 1.2).astype(int)

# Dynamic stop loss with volatility adjustment
df['adaptive_stop_loss'] = df['stop_loss_pct'].copy()

# Tighter stops in high volatility regimes when volatility is increasing
df.loc[(df['market_regime'].str.contains('High Vol', na=False)) & 
       (df['volatility_ratio'] > 1.2), 'adaptive_stop_loss'] = df['stop_loss_pct'] * 0.8

# Wider stops in low volatility regimes to avoid being whipsawed
df.loc[df['market_regime'].str.contains('Low Vol', na=False), 'adaptive_stop_loss'] = df['stop_loss_pct'] * 1.2

# Set a minimum stop loss level
df['adaptive_stop_loss'] = df['adaptive_stop_loss'].clip(0.01, 0.07)

# ------------------------------------------------------
# 2. Add Time-based Profit Taking
# ------------------------------------------------------

# Progressive profit targets based on holding period
df['holding_days'] = 0
df['profit_target'] = 0.05  # Default 5% profit target

# Use a tiered profit-taking approach
# 1-3 days: keep initial position size
# 4-7 days: reduce position by 25% if in profit
# 8+ days: reduce position by 50% if in profit > 2%

# ------------------------------------------------------
# 3. Enhanced Position Sizing Based on Conviction
# ------------------------------------------------------

# Use machine learning signal probability to adjust position size
df['signal_conviction'] = df['ml_signal_prob'] if 'ml_signal_prob' in df.columns else 0.6

# Scale position size by conviction level (0.5-1.0 range)
df['conviction_multiplier'] = (df['signal_conviction'] - 0.5) * 2  # Rescale from 0.5-1.0 to 0-1.0
df['conviction_multiplier'] = df['conviction_multiplier'].clip(0, 1) + 0.5  # 0.5 to 1.5 range

# Combine regime sizing with conviction sizing
df['improved_position_size'] = df['position_size'] * df['conviction_multiplier']

# ------------------------------------------------------
# 4. Market Regime-specific Entry Rules
# ------------------------------------------------------

# Stricter conditions for entering in challenging regimes
df['improved_signal'] = df['enhanced_signal'].copy()

# In Low Vol / Downtrend (worst regime), require higher conviction
df.loc[(df['market_regime'] == 'Low Vol / Downtrend') & 
       (df['signal_conviction'] < 0.75), 'improved_signal'] = 0

# In High Vol / Uptrend, be more aggressive
df.loc[(df['market_regime'] == 'High Vol / Uptrend') & 
       (df['signal_conviction'] > 0.6), 'improved_signal'] = 1

# Final signal with improved position sizing
df['improved_final_signal'] = df['improved_signal'] * df['improved_position_size']

# ------------------------------------------------------
# 5. Add Counter-trend Filter for Downtrends
# ------------------------------------------------------

# If in a downtrend, wait for a reversal signal
df['is_downtrend'] = (df['trend_10d'] < 0).astype(int)
df['reversal_signal'] = ((df['return_3d'] < 0) & (df['return_1d'] > 0)).astype(int)

# In downtrends, look for reversal signals
df.loc[(df['is_downtrend'] == 1) & (df['reversal_signal'] == 0) & 
       (df['market_regime'].str.contains('Downtrend', na=False)), 'improved_final_signal'] = 0

# ------------------------------------------------------
# 6. Recalculate Portfolio Performance with V2 Enhancements
# ------------------------------------------------------

# Reset portfolio for enhanced strategy v2
df['v2_portfolio'] = 1.0
current_position = 0
entry_price = 0
entry_date = 0
stop_level = 0
profit_target = 0

# Simulate trading with improved signals, stop losses, and profit targets
for i in range(1, len(df)):
    prev_day = df.iloc[i-1]
    current_day = df.iloc[i]
    
    # Update portfolio based on previous position
    if current_position > 0:  # If we had a position
        daily_return = current_day['target'] * current_position
        df.loc[df.index[i], 'v2_portfolio'] = prev_day['v2_portfolio'] * (1 + daily_return)
        
        # Calculate days in position
        holding_period = i - entry_date
        df.loc[df.index[i], 'holding_days'] = holding_period
        
        # Adjust profit target based on holding period
        if holding_period <= 3:
            profit_target = 0.03  # Initial target
        elif holding_period <= 7:
            profit_target = 0.02  # Reduce target to lock in profits faster
        else:
            profit_target = 0.015  # Even lower target for extended holds
            
        # Calculate cumulative return since entry
        cumulative_return = (current_day['close'] / entry_price - 1) if 'close' in df.columns else daily_return
        
        # Check if stop loss was hit
        if current_day['target'] < -stop_level:
            current_position = 0  # Exit position
            stop_level = 0
            profit_target = 0
            entry_date = 0
        
        # Check if profit target was hit
        elif cumulative_return > profit_target:
            # Scale out - reduce position by percentage based on holding period
            if holding_period > 7:
                current_position = current_position * 0.5  # Scale out 50%
            elif holding_period > 3:
                current_position = current_position * 0.75  # Scale out 25%
            # Recalculate stop level - move to breakeven
            stop_level = 0.005  # Tight stop at breakeven
    else:
        df.loc[df.index[i], 'v2_portfolio'] = prev_day['v2_portfolio']  # No change if not invested
    
    # Update position for next day based on signal
    if current_position == 0 and current_day['improved_final_signal'] > 0:  # Enter new position
        current_position = current_day['improved_final_signal']
        stop_level = current_day['adaptive_stop_loss']
        entry_price = current_day['close'] if 'close' in df.columns else 1.0
        entry_date = i
    elif current_position > 0 and current_day['improved_final_signal'] == 0:  # Exit position on signal
        current_position = 0
        stop_level = 0
        profit_target = 0
        entry_date = 0

# Calculate enhanced strategy v2 metrics
df['v2_return'] = df['v2_portfolio'].pct_change().fillna(0)
v2_total_return = (df['v2_portfolio'].iloc[-1] / df['v2_portfolio'].iloc[0] - 1) * 100
v2_annual_return = df['v2_return'].mean() * 252 * 100
v2_annual_vol = df['v2_return'].std() * np.sqrt(252) * 100
v2_sharpe = v2_annual_return / v2_annual_vol if v2_annual_vol > 0 else 0
v2_max_dd = ((df['v2_portfolio'] / df['v2_portfolio'].cummax()) - 1).min() * 100

# Buy signals analysis for v2
v2_buy_days = df[df['improved_final_signal'] > 0]
if len(v2_buy_days) > 0:
    v2_win_rate = (v2_buy_days['target'] > 0).mean() * 100
    v2_avg_win = v2_buy_days.loc[v2_buy_days['target'] > 0, 'target'].mean() * 100
    v2_avg_loss = v2_buy_days.loc[v2_buy_days['target'] < 0, 'target'].mean() * 100
    v2_profit_factor = abs(v2_avg_win / v2_avg_loss) if v2_avg_loss != 0 else float('inf')
else:
    v2_win_rate = 0
    v2_profit_factor = 0
    v2_avg_win = 0
    v2_avg_loss = 0

# Calculate additional risk metrics for v2
v2_returns = df['v2_return'].values
v2_negative_returns = v2_returns[v2_returns < 0]
v2_positive_returns = v2_returns[v2_returns > 0]

v2_downside_deviation = np.std(v2_negative_returns) * np.sqrt(252) * 100 if len(v2_negative_returns) > 0 else 0
v2_sortino = v2_annual_return / v2_downside_deviation if v2_downside_deviation > 0 else float('inf')

# Calculate max drawdown duration
v2_dd = (df['v2_portfolio'] / df['v2_portfolio'].cummax()) - 1
v2_dd_start = np.argmax(np.maximum.accumulate(df['v2_portfolio'].values) - df['v2_portfolio'].values)
v2_dd_end = np.argmax(df['v2_portfolio'].values[v2_dd_start:]) + v2_dd_start
v2_dd_days = v2_dd_end - v2_dd_start if v2_dd_start < v2_dd_end else 0

# Calculate Calmar ratio (return/max drawdown)
v2_calmar = abs(v2_annual_return / v2_max_dd) if v2_max_dd != 0 else float('inf')

# ------------------------------------------------------
# 7. Get Enhanced V1 Metrics for Comparison
# ------------------------------------------------------

# Total return calculation
enhanced_total_return = (df['enhanced_portfolio'].iloc[-1] / df['enhanced_portfolio'].iloc[0] - 1) * 100

# Drawdown calculations
enhanced_dd = (df['enhanced_portfolio'] / df['enhanced_portfolio'].cummax()) - 1
enhanced_max_dd = enhanced_dd.min() * 100

# Return metrics
enhanced_annual_return = df['enhanced_return'].mean() * 252 * 100
enhanced_annual_vol = df['enhanced_return'].std() * np.sqrt(252) * 100

# Risk-adjusted metrics
enhanced_returns = df['enhanced_return'].values
enhanced_negative_returns = enhanced_returns[enhanced_returns < 0]
enhanced_downside_deviation = np.std(enhanced_negative_returns) * np.sqrt(252) * 100 if len(enhanced_negative_returns) > 0 else 0
enhanced_sortino = enhanced_annual_return / enhanced_downside_deviation if enhanced_downside_deviation > 0 else float('inf')
enhanced_sharpe = enhanced_annual_return / enhanced_annual_vol if enhanced_annual_vol > 0 else 0
enhanced_calmar = abs(enhanced_annual_return / enhanced_max_dd) if enhanced_max_dd != 0 else float('inf')

# Drawdown duration
enhanced_dd_start = np.argmax(np.maximum.accumulate(df['enhanced_portfolio'].values) - df['enhanced_portfolio'].values)
enhanced_dd_end = np.argmax(df['enhanced_portfolio'].values[enhanced_dd_start:]) + enhanced_dd_start
enhanced_dd_days = enhanced_dd_end - enhanced_dd_start if enhanced_dd_start < enhanced_dd_end else 0

# Enhanced V1 buy signals analysis
enhanced_buy_days = df[df['final_signal'] > 0]
if len(enhanced_buy_days) > 0:
    enhanced_win_rate = (enhanced_buy_days['target'] > 0).mean() * 100
    enhanced_avg_win = enhanced_buy_days.loc[enhanced_buy_days['target'] > 0, 'target'].mean() * 100
    enhanced_avg_loss = enhanced_buy_days.loc[enhanced_buy_days['target'] < 0, 'target'].mean() * 100
    enhanced_profit_factor = abs(enhanced_avg_win / enhanced_avg_loss) if enhanced_avg_loss != 0 else float('inf')
else:
    enhanced_win_rate = 0
    enhanced_profit_factor = 0
    enhanced_avg_win = 0
    enhanced_avg_loss = 0

# ------------------------------------------------------
# 8. Get Original Strategy Metrics for Comparison
# ------------------------------------------------------

original_total_return = (original_df['portfolio'].iloc[-1] / original_df['portfolio'].iloc[0] - 1) * 100
original_dd = (original_df['portfolio'] / original_df['portfolio'].cummax()) - 1
original_max_dd = original_dd.min() * 100

original_returns = original_df['portfolio'].pct_change().fillna(0).values
original_annual_return = np.mean(original_returns) * 252 * 100
original_annual_vol = np.std(original_returns) * np.sqrt(252) * 100
original_sharpe = original_annual_return / original_annual_vol if original_annual_vol > 0 else 0

original_negative_returns = original_returns[original_returns < 0]
original_downside_deviation = np.std(original_negative_returns) * np.sqrt(252) * 100 if len(original_negative_returns) > 0 else 0
original_sortino = original_annual_return / original_downside_deviation if original_downside_deviation > 0 else float('inf')

original_dd_start = np.argmax(np.maximum.accumulate(original_df['portfolio'].values) - original_df['portfolio'].values)
original_dd_end = np.argmax(original_df['portfolio'].values[original_dd_start:]) + original_dd_start
original_dd_days = original_dd_end - original_dd_start if original_dd_start < original_dd_end else 0

original_calmar = abs(original_annual_return / original_max_dd) if original_max_dd != 0 else float('inf')

# Original buy signals analysis
original_buy_days = original_df[original_df['combo_signal'] > 0]
original_win_rate = (original_buy_days['target'] > 0).mean() * 100
original_avg_win = original_buy_days.loc[original_buy_days['target'] > 0, 'target'].mean() * 100
original_avg_loss = original_buy_days.loc[original_buy_days['target'] < 0, 'target'].mean() * 100
original_profit_factor = abs(original_avg_win / original_avg_loss) if original_avg_loss != 0 else float('inf')

# ------------------------------------------------------
# 9. Print Comprehensive Comparison Results
# ------------------------------------------------------

print("\n===== COMPREHENSIVE STRATEGY COMPARISON =====")
print(f"{'Metric':<25} {'Original':<15} {'Enhanced V1':<15} {'Enhanced V2':<15}")
print("-" * 70)
print(f"{'Total Return (%)':<25} {original_total_return:<15.2f} {enhanced_total_return:<15.2f} {v2_total_return:<15.2f}")
print(f"{'Annual Return (%)':<25} {original_annual_return:<15.2f} {enhanced_annual_return:<15.2f} {v2_annual_return:<15.2f}")
print(f"{'Annual Volatility (%)':<25} {original_annual_vol:<15.2f} {enhanced_annual_vol:<15.2f} {v2_annual_vol:<15.2f}")
print(f"{'Sharpe Ratio':<25} {original_sharpe:<15.4f} {enhanced_sharpe:<15.4f} {v2_sharpe:<15.4f}")
print(f"{'Sortino Ratio':<25} {original_sortino:<15.4f} {enhanced_sortino:<15.4f} {v2_sortino:<15.4f}")
print(f"{'Calmar Ratio':<25} {original_calmar:<15.4f} {enhanced_calmar:<15.4f} {v2_calmar:<15.4f}")
print(f"{'Maximum Drawdown (%)':<25} {original_max_dd:<15.2f} {enhanced_max_dd:<15.2f} {v2_max_dd:<15.2f}")
print(f"{'Max DD Duration (days)':<25} {original_dd_days:<15d} {enhanced_dd_days:<15d} {v2_dd_days:<15d}")
print(f"{'Win Rate (%)':<25} {original_win_rate:<15.2f} {enhanced_win_rate:<15.2f} {v2_win_rate:<15.2f}")
print(f"{'Avg Win (%)':<25} {original_avg_win:<15.4f} {enhanced_avg_win:<15.4f} {v2_avg_win:<15.4f}")
print(f"{'Avg Loss (%)':<25} {original_avg_loss:<15.4f} {enhanced_avg_loss:<15.4f} {v2_avg_loss:<15.4f}")
print(f"{'Profit Factor':<25} {original_profit_factor:<15.4f} {enhanced_profit_factor:<15.4f} {v2_profit_factor:<15.4f}")

# ------------------------------------------------------
# 10. Visualize Results
# ------------------------------------------------------

# Plot portfolio value comparison (log scale for better visibility)
plt.figure(figsize=(12, 6))
plt.semilogy(df['date'], df['portfolio'], label='Original')
plt.semilogy(df['date'], df['enhanced_portfolio'], label='Enhanced V1')
plt.semilogy(df['date'], df['v2_portfolio'], label='Enhanced V2')
plt.title('Portfolio Performance Comparison (Log Scale)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/enhanced_v2_comparison_log.png')
print("\nLog-scale performance comparison plot saved to results/enhanced_v2_comparison_log.png")

# Regular scale
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['portfolio'], label='Original')
plt.plot(df['date'], df['enhanced_portfolio'], label='Enhanced V1')
plt.plot(df['date'], df['v2_portfolio'], label='Enhanced V2')
plt.title('Portfolio Performance Comparison')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/enhanced_v2_comparison.png')
print("Performance comparison plot saved to results/enhanced_v2_comparison.png")

# Plot drawdowns
plt.figure(figsize=(12, 6))
plt.plot(df['date'], original_dd * 100, label='Original')
plt.plot(df['date'], enhanced_dd * 100, label='Enhanced V1')
plt.plot(df['date'], v2_dd * 100, label='Enhanced V2')
plt.title('Drawdown Comparison')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/enhanced_v2_drawdown.png')
print("Drawdown comparison plot saved to results/enhanced_v2_drawdown.png")

# Monthly returns heatmap
# Group returns by month and year
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')

# Calculate returns by month for each strategy
monthly_returns = pd.DataFrame()
monthly_returns['year'] = df.groupby(['year', 'month'])['date'].first().reset_index()['year']
monthly_returns['month'] = df.groupby(['year', 'month'])['date'].first().reset_index()['month'] 
monthly_returns['month_name'] = monthly_returns['month'].apply(lambda x: {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}.get(x, ''))

# Calculate average returns by month
monthly_returns['Original'] = df.groupby(['year', 'month'])['target'].mean().reset_index()['target'] * 100
monthly_returns['Enhanced_V1'] = df.groupby(['year', 'month'])['enhanced_return'].mean().reset_index()['enhanced_return'] * 100
monthly_returns['Enhanced_V2'] = df.groupby(['year', 'month'])['v2_return'].mean().reset_index()['v2_return'] * 100

# Create better labels for the strategies
strategy_labels = {
    'Original': 'Original Strategy',
    'Enhanced_V1': 'Enhanced V1',
    'Enhanced_V2': 'Enhanced V2'
}

# Create a pivot table for the heatmap
for strategy, label in strategy_labels.items():
    pivot = pd.pivot_table(monthly_returns, values=strategy, index='month_name', columns='year')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
    plt.title(f'{label} Monthly Returns (%)')
    plt.tight_layout()
    plt.savefig(f'results/{strategy.lower()}_monthly_returns.png')
    print(f"{label} monthly returns heatmap saved to results/{strategy.lower()}_monthly_returns.png")

# Regime-based performance
plt.figure(figsize=(14, 7))
regimes = df['market_regime'].dropna().unique()

# Get performance by regime for each strategy
original_regime_returns = []
enhanced_regime_returns = []
v2_regime_returns = []

for regime in regimes:
    regime_df = df[df['market_regime'] == regime].dropna(subset=['target'])
    
    if len(regime_df) == 0:
        original_regime_returns.append(0)
        enhanced_regime_returns.append(0)
        v2_regime_returns.append(0)
        continue
    
    # Original strategy
    orig_days = regime_df[regime_df['combo_signal'] == 1]
    if len(orig_days) > 0:
        original_regime_returns.append(orig_days['target'].mean() * 100)
    else:
        original_regime_returns.append(0)
    
    # Enhanced V1 strategy
    enh_days = regime_df[regime_df['final_signal'] > 0]
    if len(enh_days) > 0:
        enhanced_regime_returns.append(enh_days['target'].mean() * 100)
    else:
        enhanced_regime_returns.append(0)
    
    # Enhanced V2 strategy
    v2_days = regime_df[regime_df['improved_final_signal'] > 0]
    if len(v2_days) > 0:
        v2_regime_returns.append(v2_days['target'].mean() * 100)
    else:
        v2_regime_returns.append(0)

# Plot the comparison only if we have valid regimes
if len(regimes) > 0:
    x = np.arange(len(regimes))
    width = 0.25
    
    plt.bar(x - width, original_regime_returns, width, label='Original')
    plt.bar(x, enhanced_regime_returns, width, label='Enhanced V1')
    plt.bar(x + width, v2_regime_returns, width, label='Enhanced V2')
    plt.title('Returns by Market Regime')
    plt.xlabel('Market Regime')
    plt.ylabel('Average Return (%)')
    plt.xticks(x, regimes, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('results/regime_strategy_v2_comparison.png')
    print("Regime performance comparison plot saved to results/regime_strategy_v2_comparison.png")
else:
    print("No valid market regimes found for comparison.")

# Save the enhanced V2 strategy results
df.to_csv('results/enhanced_v2_strategy_results.csv', index=False)
print("\nEnhanced V2 strategy results saved to results/enhanced_v2_strategy_results.csv") 