
# Trading Algorithm Performance Report
**Generated:** 2025-04-06 16:06:48
**Data Period:** 2024-07-02 to 2025-03-16 (607 days)

## 1. Overall Performance Metrics

| Metric | Value |
|--------|-------|
| Starting Value | $1.00 |
| Final Value | $12.20 |
| Total Return | 1119.50% |
| Annualized Return | 349.93% |
| Annualized Volatility | 155.55% |
| Sharpe Ratio | 2.2496 |
| Maximum Drawdown | -48.18% |

## 2. Signal Analysis

| Signal Type | Count | Percentage | Win Rate | Average Return |
|-------------|-------|------------|----------|----------------|
| Buy Signals | 304 | 50.1% | 61.8% | 2.0260% |
| Sell/Hold Signals | 303 | 49.9% | 53.8% | -1.1576% |

**Profit Factor:** 2.3184

**Average Win/Loss Ratio:** 1.4305

## 3. Market Regime Performance


### Low Vol / Uptrend
- **Days:** 200 (32.9% of total)
- **Buy Signals:** 108 (54.0% of regime days)
- **Market Return:** 1.4152%
- **Strategy Return:** 2.3959%
- **Win Rate:** 62.0%
- **Outperformance:** 0.9807%

### Low Vol / Downtrend
- **Days:** 101 (16.6% of total)
- **Buy Signals:** 43 (42.6% of regime days)
- **Market Return:** -1.1964%
- **Strategy Return:** -1.4538%
- **Win Rate:** 41.9%
- **Outperformance:** -0.2574%

### High Vol / Uptrend
- **Days:** 167 (27.5% of total)
- **Buy Signals:** 85 (50.9% of regime days)
- **Market Return:** 2.3765%
- **Strategy Return:** 3.6788%
- **Win Rate:** 64.7%
- **Outperformance:** 1.3023%

### High Vol / Downtrend
- **Days:** 130 (21.4% of total)
- **Buy Signals:** 59 (45.4% of regime days)
- **Market Return:** -2.2565%
- **Strategy Return:** 1.8228%
- **Win Rate:** 71.2%
- **Outperformance:** 4.0793%

## 4. Signal Stability

| Metric | Value |
|--------|-------|
| Buy-to-Sell Transitions | 32 |
| Sell-to-Buy Transitions | 31 |
| Avg Buy Signal Duration | 9.50 days |
| Avg Sell/Hold Signal Duration | 9.47 days |

### Quick Signal Reversals (Buy→Sell→Buy)
- **Within 1 days:** 4 (12.5% of all buy-to-sell transitions)
- **Within 2 days:** 6 (18.8% of all buy-to-sell transitions)
- **Within 3 days:** 8 (25.0% of all buy-to-sell transitions)
- **Within 5 days:** 13 (40.6% of all buy-to-sell transitions)
- **Within 7 days:** 13 (40.6% of all buy-to-sell transitions)
- **Within 10 days:** 16 (50.0% of all buy-to-sell transitions)

## 5. Monthly Performance

| Month | Market Return | Strategy Return | Win Rate | Outperformance |
|-------|---------------|-----------------|----------|----------------|
| Jul 2024 | 0.01% | 0.01% | 62.1% | 0.00% |
| Aug 2024 | 0.15% | 0.15% | 55.2% | 0.00% |
| Sep 2024 | 0.26% | 0.41% | 50.0% | 0.14% |
| Oct 2024 | -0.30% | 1.44% | 65.3% | 1.74% |
| Nov 2024 | -0.06% | 2.11% | 63.6% | 2.17% |
| Dec 2024 | 1.23% | 1.51% | 50.0% | 0.28% |
| Jan 2025 | 0.46% | 3.37% | 71.4% | 2.90% |
| Feb 2025 | 1.31% | 9.34% | 80.8% | 8.03% |
| Mar 2025 | -1.02% | 0.30% | 40.0% | 1.33% |

## 6. Day of Week Performance

| Day | Market Return | Strategy Return | Win Rate | Outperformance |
|-----|---------------|-----------------|----------|----------------|
| Monday | -0.15% | 1.87% | 63.3% | 2.02% |
| Tuesday | 1.86% | 1.86% | 62.7% | 0.00% |
| Wednesday | 1.95% | 2.42% | 60.4% | 0.46% |
| Thursday | -1.78% | 0.68% | 58.1% | 2.46% |
| Friday | 1.10% | 0.55% | 53.3% | -0.55% |
| Saturday | 0.51% | 5.31% | 73.5% | 4.80% |
| Sunday | -0.37% | 1.24% | 58.6% | 1.60% |

## 7. Significant Drawdown Periods

| Period | Duration | Max Drawdown | Most Common Regime |
|--------|----------|--------------|--------------------|
| 2024-07-10 to 2024-07-11 | 1 days | -10.43% | Low Vol / Uptrend (50.0%) |
| 2024-08-01 to 2024-10-29 | 89 days | -48.18% | Low Vol / Uptrend (51.1%) |
| 2024-11-12 to 2024-11-21 | 9 days | -19.44% | High Vol / Uptrend (58.1%) |
| 2024-11-30 to 2024-12-02 | 2 days | -11.78% | High Vol / Downtrend (100.0%) |
| 2024-12-03 to 2024-12-25 | 22 days | -21.65% | Low Vol / Uptrend (44.7%) |
| 2025-01-25 to 2025-01-28 | 3 days | -14.54% | High Vol / Uptrend (53.3%) |
| 2025-03-05 to 2025-03-09 | 4 days | -10.68% | High Vol / Uptrend (60.0%) |
| 2025-03-11 to 2025-03-16 | 5 days | -34.93% | High Vol / Downtrend (100.0%) |

## 8. Key Findings and Recommendations

1. **Overall Performance**: The algorithm has demonstrated strong overall performance with a significant total return and positive Sharpe ratio, indicating good risk-adjusted returns.

2. **Market Regime Analysis**: 
   - The strategy performs best in high volatility downtrend environments, showing exceptional outperformance.
   - It struggles most in low volatility downtrend environments, which is the only regime where it underperforms the market.

3. **Signal Stability**: 
   - The algorithm tends to maintain signals for reasonable periods, with buy signals lasting longer on average than sell signals.
   - There are some quick reversals in signals, which could potentially be filtered to reduce transaction costs.

4. **Seasonality**: 
   - Monthly performance shows strong results in some months and weaker results in others, suggesting potential seasonal patterns.
   - Day of week analysis reveals certain days consistently outperform others.

5. **Recommendations**:
   - Consider implementing a signal filter to prevent quick reversals and reduce potential transaction costs.
   - Potentially adjust position sizing based on market regimes, increasing exposure in high volatility downtrends and decreasing in low volatility downtrends.
   - Further investigate seasonal patterns to potentially optimize entry/exit timing.
   - Monitor drawdown periods closely, as they tend to occur most frequently in specific market regimes.
