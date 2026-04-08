import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# =============================================================================
# 1. Data Loading and Cleaning
# =============================================================================

def load_and_clean_carry_data(filepath='script/course/Carry.xlsx'):
    """
    读取Carry.xlsx文件，转换日期为datetime格式，数值为float格式
    """
    xls = pd.ExcelFile(filepath)
    data_dict = {}
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet)
        
        # Convert Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
        
        # Convert numeric columns (remove % and divide by 100)
        for col in df.columns:
            if col != 'Date' and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        
        data_dict[sheet] = df
        print(f"✓ Loaded {sheet}: {df.shape[0]} rows, {df.shape[1]} columns")
    for key in data_dict.keys():
        print(f"\nPreview of '{key}' sheet:")
        print(data_dict[key].head())
        print(data_dict[key].info())
    return data_dict


# =============================================================================
# 2. Currency Ranking Analysis
# =============================================================================

def compute_currency_ranks(interest_rate_df):
    """
    For each month, rank countries by interest rate (1 = lowest rate)
    Returns: DataFrame with ranks, and average ranks by country
    """
    df = interest_rate_df.copy()
    countries = [c for c in df.columns if c != 'Date']
    
    # Compute monthly ranks (1 = lowest interest rate)
    rank_df = df[['Date']].copy()
    for col in countries:
        # Rank within each month: lowest rate gets rank 1
        rank_df[col] = df.groupby('Date')[col].rank(method='min', ascending=True)
    
    # Compute average rank for each country
    avg_ranks = rank_df[countries].mean().sort_values()
    print("\n✓ Computed monthly ranks and average ranks for each country:")
    print(rank_df.head())
    return rank_df, avg_ranks


def answer_ranking_questions(avg_ranks):
    """
    Answer the ranking questions from the homework
    """
    print("\n" + "="*70)
    print("📊 CURRENCY RANKING RESULTS")
    print("="*70)
    
    # Q1: Lowest average rank = most often funding currency
    funding_1 = avg_ranks.idxmin()
    funding_1_rank = avg_ranks.min()
    print(f"\nQ1. Most often funding currency (lowest avg rank):")
    print(f"    🏦 {funding_1}: Average Rank = {funding_1_rank:.3f}")
    
    # Q2: Second lowest average rank
    funding_2 = avg_ranks.index[1]
    funding_2_rank = avg_ranks.iloc[1]
    print(f"\nQ2. Second most often funding currency:")
    print(f"    🏦 {funding_2}: Average Rank = {funding_2_rank:.3f}")
    
    # Q3: Highest average rank = most often investment currency
    invest_1 = avg_ranks.idxmax()
    invest_1_rank = avg_ranks.max()
    print(f"\nQ3. Most often investment currency (highest avg rank):")
    print(f"    💰 {invest_1}: Average Rank = {invest_1_rank:.3f}")
    
    # Q4: Second highest average rank
    invest_2 = avg_ranks.index[-2]
    invest_2_rank = avg_ranks.iloc[-2]
    print(f"\nQ4. Second most often investment currency:")
    print(f"    💰 {invest_2}: Average Rank = {invest_2_rank:.3f}")
    
    print("\n" + "-"*70)
    print("Average Ranks for All Countries (1=lowest rate, 9=highest):")
    print("-"*70)
    for rank, (country, avg_rank) in enumerate(avg_ranks.items(), 1):
        marker = "🏦" if rank <= 2 else ("💰" if rank >= 8 else "")
        print(f"  {rank}. {country:15s}: {avg_rank:6.3f} {marker}")
    
    return funding_1, funding_2, invest_1, invest_2


def plot_us_rank_over_time(rank_df, save_path='us_rank_plot.png'):
    """
    Plot the rank of the US over time
    """
    plt.figure(figsize=(12, 5))
    plt.plot(rank_df['Date'], rank_df['US'], linewidth=2, color='navy')
    plt.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Lowest Rate (Rank=1)')
    plt.axhline(y=9, color='red', linestyle='--', alpha=0.3, label='Highest Rate (Rank=9)')
    plt.axhline(y=5, color='gray', linestyle=':', alpha=0.5, label='Median Rank')
    
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('US Interest Rate Rank (1=Lowest, 9=Highest)', fontsize=11)
    plt.title('US Currency Rank Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ US rank plot saved to '{save_path}'")
    plt.show()


# =============================================================================
# 3. Carry Trade Position Construction
# =============================================================================

def create_carry_positions(interest_rate_df, n_long_short=3):
    """
    Create positions: short n currencies with lowest rates, long n with highest rates
    
    Returns: DataFrame with positions (-1, 0, +1) for each currency
    """
    df = interest_rate_df.copy()
    countries = [c for c in df.columns if c != 'Date']
    
    positions_df = df[['Date']].copy()
    
    for idx, row in df.iterrows():
        # Get rates for this month (excluding Date)
        rates = row[countries].dropna()
        
        # Rank currencies by rate
        ranked = rates.rank(method='min', ascending=True)
        
        # Initialize positions to 0
        positions = pd.Series(0, index=countries)
        
        # Short the n lowest rate currencies (funding)
        lowest_n = ranked.nsmallest(n_long_short).index
        positions[lowest_n] = -1
        
        # Long the n highest rate currencies (investment)
        highest_n = ranked.nlargest(n_long_short).index
        positions[highest_n] = +1
        
        positions_df.loc[idx, countries] = positions
    
    return positions_df


def answer_canada_position_question(positions_df):
    """
    Q6: What is the average position in Canada?
    """
    avg_position_canada = positions_df['Canada'].mean()
    print(f"\nQ6. Average position in Canada: {avg_position_canada:+.4f}")
    print(f"    Interpretation: {'Long bias' if avg_position_canada > 0 else 'Short bias' if avg_position_canada < 0 else 'Neutral'}")
    return avg_position_canada


# =============================================================================
# 4. Portfolio Return Calculation
# =============================================================================

def compute_portfolio_returns(positions_df, exret_df):
    """
    Compute portfolio excess returns with correct timing:
    - Positions at time t are based on rates at time t
    - Returns are realized from t to t+1
    
    Portfolio return at t+1 = sum(positions_t * excess_returns_t+1)
    """
    # Merge positions and excess returns on Date
    merged = positions_df.merge(exret_df, on='Date', suffixes=('_pos', '_ret'))
    
    countries = [c for c in positions_df.columns if c != 'Date']
    
    # Compute portfolio return for each month
    # Note: positions determined at beginning of month, returns realized during month
    portfolio_returns = []
    dates = []
    
    for idx in range(len(merged) - 1):
        # Positions from current row
        positions = merged.loc[idx, countries].values.astype(float)
        # Returns from NEXT row (realized during the month)
        returns = merged.loc[idx + 1, [f'{c}_ret' for c in countries]].values.astype(float)
        
        # Portfolio return = sum(w_i * r_i)
        port_ret = np.nansum(positions * returns)
        portfolio_returns.append(port_ret)
        dates.append(merged.loc[idx + 1, 'Date'])
    
    result_df = pd.DataFrame({'Date': dates, 'Portfolio_Return': portfolio_returns})
    return result_df


def compute_performance_metrics(returns_df, freq='monthly'):
    """
    Compute annualized performance metrics
    """
    returns = returns_df['Portfolio_Return'].dropna()
    
    # Basic statistics
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    # Annualization factors
    if freq == 'monthly':
        ann_factor = 12
        sqrt_factor = np.sqrt(12)
    else:
        ann_factor = 252
        sqrt_factor = np.sqrt(252)
    
    # Annualized metrics
    ann_mean = mean_ret * ann_factor
    ann_std = std_ret * sqrt_factor
    sharpe = ann_mean / ann_std if ann_std > 0 else np.nan
    
    # Monthly skewness
    skewness = stats.skew(returns)
    
    print("\n" + "="*70)
    print("📈 PORTFOLIO PERFORMANCE METRICS")
    print("="*70)
    print(f"\nSample period: {returns_df['Date'].min().date()} to {returns_df['Date'].max().date()}")
    print(f"Number of observations: {len(returns)}")
    print(f"\nQ7. Annualized average excess return: {ann_mean:+.4f} ({ann_mean*100:+.2f}%)")
    print(f"Q8. Annualized standard deviation:   {ann_std:+.4f} ({ann_std*100:+.2f}%)")
    print(f"Q9. Annualized Sharpe Ratio:         {sharpe:+.4f}")
    print(f"Q10. Monthly skewness of returns:     {skewness:+.4f}")
    
    # Interpretation
    print(f"\n📝 Interpretation:")
    if sharpe > 0.5:
        print(f"   ✓ Positive risk-adjusted returns (Sharpe > 0.5)")
    elif sharpe > 0:
        print(f"   ⚠ Modest risk-adjusted returns (0 < Sharpe < 0.5)")
    else:
        print(f"   ✗ Negative risk-adjusted returns (Sharpe < 0)")
    
    if skewness < 0:
        print(f"   ⚠ Negative skew: carry trade has crash risk (left tail)")
    elif skewness > 0:
        print(f"   ✓ Positive skew: favorable return distribution")
    
    return {
        'annualized_mean': ann_mean,
        'annualized_std': ann_std,
        'sharpe_ratio': sharpe,
        'monthly_skewness': skewness,
        'n_obs': len(returns)
    }


def plot_portfolio_returns(returns_df, save_path='portfolio_returns.png'):
    """
    Plot cumulative portfolio returns and histogram
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cumulative returns
    returns = returns_df['Portfolio_Return'].dropna()
    cumulative = (1 + returns).cumprod() - 1
    
    axes[0].plot(returns_df['Date'].iloc[1:], cumulative, linewidth=2, color='navy')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Cumulative Excess Return')
    axes[0].set_title('Cumulative Portfolio Returns')
    axes[0].grid(True, alpha=0.3)
    
    # Return distribution
    axes[1].hist(returns, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(x=returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean()*100:.2f}%')
    axes[1].set_xlabel('Monthly Excess Return')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Monthly Returns')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Portfolio returns plot saved to '{save_path}'")
    plt.show()


# =============================================================================
# 5. Main Execution
# =============================================================================

def main():
    print("🚀 Starting Currency Carry Trade Backtest")
    print("="*70)
    
    # Step 1: Load and clean data
    print("\n[1/5] Loading data...")
    data = load_and_clean_carry_data('script/course/Carry.xlsx')
    print(f"\nData sheets loaded: {list(data.keys())}")
    print(data)
    interest_rate_df = data['InterestRate']
    exret_df = data['ExRet']
    
    # Step 2: Currency ranking analysis
    print("\n[2/5] Computing currency rankings...")
    rank_df, avg_ranks = compute_currency_ranks(interest_rate_df)
    funding_1, funding_2, invest_1, invest_2 = answer_ranking_questions(avg_ranks)
    
    # Step 3: Plot US rank over time
    # print("\n[3/5] Plotting US rank over time...")
    # plot_us_rank_over_time(rank_df)
    
    # # Step 4: Create carry trade positions
    # print("\n[4/5] Creating carry trade positions (short 3 lowest, long 3 highest)...")
    # positions_df = create_carry_positions(interest_rate_df, n_long_short=3)
    # avg_canada_pos = answer_canada_position_question(positions_df)
    
    # # Step 5: Compute portfolio returns and performance metrics
    # print("\n[5/5] Computing portfolio returns and performance metrics...")
    # portfolio_returns_df = compute_portfolio_returns(positions_df, exret_df)
    # metrics = compute_performance_metrics(portfolio_returns_df)
    
    # Plot portfolio returns
    plot_portfolio_returns(portfolio_returns_df)
    
    # Summary table for LaTeX export
    print("\n" + "="*70)
    print("📋 RESULTS SUMMARY (LaTeX-ready)")
    print("="*70)
    # print(f"""



if __name__ == "__main__":
    results = main()