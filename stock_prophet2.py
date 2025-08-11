# -*- coding: utf-8 -*-
"""
Prophet-Based Stock Analysis Tool

This script fetches historical stock data, performs a time-series forecast
using Meta's Prophet library, and provides a buy/sell/hold signal along with
a risk assessment and optional portfolio projection.

Refactored to incorporate best practices:
- Uses `yfinance` for robust data fetching and dividend information.
- Uses `argparse` for a professional command-line interface.
- Adds U.S. holidays to the model for improved accuracy.
- Visualizes the forecast using `matplotlib`.
- Defines configuration and thresholds as constants.
- Adds data scaling to prevent numerical instability warnings.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from prophet import Prophet
import warnings
import argparse
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Suppress informational messages from Prophet and runtime warnings from numpy
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# #################################
# ##     CONFIGURATION & THRESHOLDS   ##
# #################################

# How far back to get historical data from. Format: 'YYYY-MM-DD'
START_DATE_STR = '2023-01-01'

# How many days into the future you want to forecast.
FORECAST_DAYS = 90

# --- Analysis Thresholds (Moved here from being "magic numbers") ---
YIELD_THRESHOLD_GOOD = 4.0
YIELD_THRESHOLD_AVG = 2.0
SIGNAL_PRICE_CHANGE_PCT = 5.0
RISK_METRIC_LOW = 5.0
RISK_METRIC_MEDIUM = 15.0
RISK_METRIC_HIGH = 25.0

# --- Plotting Configuration ---
# Set to False to disable plots by default. Can be overridden with --no-plots flag.
SHOW_PLOTS = True


# #############################
# ##     ANALYSIS SCRIPT     ##
# #############################

def get_stock_data(ticker: str, start_date_str: str) -> Tuple[
    Optional[pd.DataFrame], str, Optional[pd.Series], float, str, float, float]:
    """
    Fetches daily stock data, company name, and dividend information using yfinance.

    Args:
        ticker (str): The stock ticker symbol.
        start_date_str (str): The start date for historical data.

    Returns:
        A tuple containing:
        - DataFrame of historical prices.
        - The stock's long name.
        - A Series of historical dividends for the period.
        - The trailing annual dividend rate for projections.
        - The currency of the stock.
        - The 52-week high.
        - The 52-week low.
    """
    try:
        # Use the yfinance Ticker object
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        stock_name = stock_info.get('longName', ticker)
        currency = stock_info.get('currency', '$')
        fifty_two_week_high = stock_info.get('fiftyTwoWeekHigh', 0.0)
        fifty_two_week_low = stock_info.get('fiftyTwoWeekLow', 0.0)

        # Get the trailing annual dividend rate for projections. Default to 0 if not available.
        baseline_dividend = stock_info.get('trailingAnnualDividendRate', 0.0)

        # Fetch history. auto_adjust=True handles stock splits and dividends.
        df = stock.history(start=start_date_str, auto_adjust=True)

        # Fetch historical dividends for the same period
        dividends = stock.dividends[start_date_str:]

        if df.empty:
            print(f"⚠️ No data found for '{ticker}' for the specified period.")
            return None, ticker, None, 0.0, currency, 0.0, 0.0

        # Prepare dataframe for Prophet: needs 'ds' and 'y' columns.
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        # Ensure 'ds' is timezone-naive for Prophet compatibility
        df['ds'] = df['ds'].dt.tz_localize(None)

        # Clean data: remove any rows where the price is missing or non-positive
        df.dropna(subset=['y'], inplace=True)
        if (df['y'] <= 0).any():
            print(f"⚠️ Warning: Stock data for {ticker} contains zero/negative prices. Removing them.")
            df = df[df['y'] > 0]

        start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
        formatted_start_date = start_date_obj.strftime('%d-%b-%Y')
        print(
            f"✅ Successfully fetched {len(df)} days of data (from {formatted_start_date}) for {stock_name} ({ticker}).")

        return df, stock_name, dividends, baseline_dividend, currency, fifty_two_week_high, fifty_two_week_low

    except Exception as e:
        print(f"❌ An unexpected error occurred while fetching data for '{ticker}': {e}")
        return None, ticker, None, 0.0, '$', 0.0, 0.0


def analyze_and_forecast(df: pd.DataFrame, display_name: str, forecast_days: int, baseline_dividend: float,
                         historical_dividends: pd.Series, show_plots: bool, currency: str, fifty_two_week_high: float,
                         fifty_two_week_low: float) -> Tuple[Optional[pd.DataFrame], str, str, str, str, float]:
    """
    Performs Prophet forecasting, prints analysis, and optionally visualizes the result.
    Returns the forecast and key analysis metrics.
    """
    # --- Data Transformation & Scaling ---
    # Step 1: Log transform the price to stabilize variance.
    df['y_log'] = np.log(df['y'])

    # Step 2: Scale the log-transformed data to a [0, 1] range.
    # This prevents numerical errors in Prophet's backend (Stan).
    y_log_min = df['y_log'].min()
    y_log_max = df['y_log'].max()
    df['y_scaled'] = (df['y_log'] - y_log_min) / (y_log_max - y_log_min)

    # --- Model Training ---
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_country_holidays(country_name='US')  # Assuming US market holidays

    # Fit the model on the scaled data
    model.fit(df[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'}))

    # --- Forecasting ---
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # --- Inverse Transform & Analysis ---
    # The forecast is on the scaled [0, 1] range. We must reverse the transformations.

    # Step 1: Inverse the min-max scaling to get back to the log scale
    for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
        if col in forecast.columns:
            forecast[col] = forecast[col] * (y_log_max - y_log_min) + y_log_min

    # Step 2: Inverse the log transform (exponentiate) to get the original price scale
    current_price = df['y'].iloc[-1]
    future_prediction = forecast.iloc[-1]
    predicted_price = np.exp(future_prediction['yhat'])
    predicted_lower = np.exp(future_prediction['yhat_lower'])
    predicted_upper = np.exp(future_prediction['yhat_upper'])

    # --- Generate Analysis Report ---
    signal, reason, risk_score, yield_status, projected_yield = print_analysis_report(
        df, forecast, display_name, current_price, predicted_price,
        predicted_lower, predicted_upper, baseline_dividend, historical_dividends, currency, fifty_two_week_high,
        fifty_two_week_low
    )

    # --- Visualization (Conditional) ---
    if show_plots:
        print("\n📈 Generating forecast plots...")
        try:
            # Custom plot to show final forecast in the original price scale
            fig, ax = plt.subplots(figsize=(12, 7))

            # Plot historical data in original scale
            ax.plot(df['ds'], df['y'], 'k.', label='Historical Price')

            # Plot forecast in original scale
            ax.plot(forecast['ds'], np.exp(forecast['yhat']), ls='-', c='#0072B2', label='Forecast')

            # Plot uncertainty interval in original scale
            ax.fill_between(
                forecast['ds'],
                np.exp(forecast['yhat_lower']),
                np.exp(forecast['yhat_upper']),
                color='#0072B2',
                alpha=0.2,
                label='80% Confidence Interval'
            )

            ax.set_title(f'Price Forecast for {display_name}', size=16)
            ax.set_xlabel("Date", size=12)
            ax.set_ylabel(f"Price ({currency})", size=12)
            ax.legend()
            ax.grid(True, which='both', ls='--', lw=0.5)
            plt.show()

            # The components plot is still useful and is plotted on the log scale
            model.plot_components(forecast)
            plt.show()

        except Exception as e:
            print(f"Could not generate plots. Error: {e}")

    # Return the forecast with prices in their original scale for portfolio analysis
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
    return forecast, signal, reason, risk_score, yield_status, projected_yield


def print_analysis_report(df, forecast, display_name, current_price, predicted_price, predicted_lower, predicted_upper,
                          baseline_dividend, historical_dividends, currency, fifty_two_week_high, fifty_two_week_low) -> \
Tuple[str, str, str, str, float]:
    """
    Prints a formatted analysis report to the console and returns key metrics.
    """
    last_known_date_obj = df['ds'].iloc[-1]
    predicted_date_obj = forecast['ds'].iloc[-1]

    # --- Projected Dividend Yield Calculation ---
    projected_yield = (baseline_dividend / predicted_price) * 100 if predicted_price > 0 else 0
    if projected_yield > YIELD_THRESHOLD_GOOD:
        yield_status = "Good"
    elif projected_yield > YIELD_THRESHOLD_AVG:
        yield_status = "Average"
    else:
        yield_status = "Low"

    # --- Buy/Sell Signal Logic ---
    price_change_pct = ((predicted_price - current_price) / current_price) * 100
    if price_change_pct > SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📈 BUY", f"Price is forecast to rise by {price_change_pct:.2f}%."
    elif price_change_pct < -SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📉 SELL", f"Price is forecast to fall by {price_change_pct:.2f}%."
    else:
        signal, reason = "🤔 HOLD", f"Price is forecast to be stable (change: {price_change_pct:.2f}%)."

    # --- Risk Assessment ---
    uncertainty_range = predicted_upper - predicted_lower
    risk_metric = (uncertainty_range / predicted_price) * 100
    if risk_metric < RISK_METRIC_LOW:
        risk_score = "1/5 (Very Low)"
    elif risk_metric < RISK_METRIC_MEDIUM:
        risk_score = "2/5 (Low)"
    elif risk_metric < RISK_METRIC_HIGH:
        risk_score = "3/5 (Medium)"
    else:
        risk_score = "4/5 (High)"

    # --- Print to Console ---
    analysis_time = datetime.now(ZoneInfo("Asia/Tokyo")).strftime('%d-%b-%Y %I:%M:%S %p %Z')
    header = f"PROPHET-BASED STOCK ANALYSIS: {display_name}"
    print("\n" + "=" * 70)
    print(header.center(70))
    print(f"Analysis Time: {analysis_time}".center(70))
    print("=" * 70)
    print(f"Forecasting {FORECAST_DAYS} days (until {predicted_date_obj.strftime('%d-%b-%Y')}).\n")

    print(f"🔹 Last Market Data (as of {last_known_date_obj.strftime('%d-%b-%Y')}):")
    print(f"   - Closing Price: {currency} {current_price:,.2f}")

    # --- Historical High/Low Information ---
    print(f"   - Historical High / Low:")
    # Group the dataframe by year to find the high/low for each year
    df_by_year = df.groupby(df['ds'].dt.year)

    for year, year_df in df_by_year:
        # Find the row with the highest price for the year to get the date
        high_row = year_df.loc[year_df['High'].idxmax()]
        year_high_price = high_row['High']
        year_high_date = high_row['ds'].strftime('%d-%b-%Y')

        # Find the row with the lowest price for the year to get the date
        low_row = year_df.loc[year_df['Low'].idxmin()]
        year_low_price = low_row['Low']
        year_low_date = low_row['ds'].strftime('%d-%b-%Y')

        print(
            f"     - {year} High / Low: {currency} {year_high_price:,.2f} ({year_high_date}) / {currency} {year_low_price:,.2f} ({year_low_date})")

    print(f"\n   - 52-Week High / Low: {currency} {fifty_two_week_high:,.2f} / {currency} {fifty_two_week_low:,.2f}\n")

    print(f"🔮 Forecast for {predicted_date_obj.strftime('%d-%b-%Y')}:")
    print(f"   - Predicted Price: {currency} {predicted_price:,.2f}")
    print(f"   - Expected Range: {currency} {predicted_lower:,.2f} to {currency} {predicted_upper:,.2f}\n")

    print(f"💰 Dividend Yield Analysis:")
    print(
        f"   - Projected Yield: {projected_yield:.2f}% ({currency} {baseline_dividend:.4f} per share) (based on a predicted price of {currency} {predicted_price:,.2f})")
    print(f"   - Projection Status: {yield_status}")

    # --- Historical Dividend Information ---
    if historical_dividends is not None and not historical_dividends.empty:
        dividends_by_year = historical_dividends.groupby(historical_dividends.index.year)
        if not dividends_by_year.groups:
            print("   - Historical Payout: No dividends were paid in the tracked period.\n")
        else:
            print("   - Historical Payouts:")
            for year, dividends_in_year in dividends_by_year:
                total_dividends_for_year = dividends_in_year.sum()
                historical_yield_pct = (total_dividends_for_year / current_price) * 100 if current_price > 0 else 0
                print(
                    f"     - {year} Payout: {historical_yield_pct:.2f}% ({currency} {total_dividends_for_year:.4f} per share)")
            print()  # Add a newline for spacing
    else:
        print("   - Historical Payout: No dividends were paid in the tracked period.\n")

    print(f"📊 Buy/Sell Signal:")
    print(f"   - Recommendation: {signal}")
    print(f"   - Justification: {reason}\n")

    print(f"⚠️ Risk Assessment:")
    print(f"   - Model Uncertainty ({risk_metric:.2f}%) suggests the risk is: {risk_score}")
    print("=" * 70)

    return signal, reason, risk_score, yield_status, projected_yield


def display_portfolio_analysis(forecast: pd.DataFrame, buy_price: float, units: int, signal: str, reason: str,
                               currency: str):
    """
    Calculates and displays a personalized portfolio projection with an action signal.
    """
    predicted_price = forecast.iloc[-1]['yhat']
    predicted_date = forecast.iloc[-1]['ds'].date().strftime('%d-%b-%Y')

    initial_cost = buy_price * units
    predicted_value = predicted_price * units
    potential_pl = predicted_value - initial_cost
    potential_pl_percent = (potential_pl / initial_cost) * 100 if initial_cost > 0 else 0

    # --- Determine Portfolio Action and Justification ---
    if "BUY" in signal:
        action_signal = "📈 Accumulate More"
        justification = f"The forecast is positive ({reason}) and your position is projected to grow by {potential_pl_percent:+.2f}%. This could be a good opportunity to increase your holdings."
    elif "SELL" in signal:
        action_signal = "📉 Secure Profits / Cut Losses"
        if potential_pl > 0:
            justification = f"The forecast is negative ({reason}). Consider selling to secure your current profit of {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
        else:
            justification = f"The forecast is negative ({reason}). Consider selling to prevent further losses on your position, currently at {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
    else:  # HOLD
        action_signal = "🤔 Hold Position"
        justification = f"The forecast is stable. Your position's projected gain is {potential_pl_percent:+.2f}%. It may be best to monitor the stock for new developments."

    print("\n" + "---" * 20)
    print("💼               PORTFOLIO PROJECTION")
    print("---" * 20)
    print(f"Based on your input of {units} units at {currency} {buy_price:,.2f} each.\n")
    print(f"   - Initial Investment: {currency} {initial_cost:,.2f}")
    print(f"   - Predicted Value on {predicted_date}: {currency} {predicted_value:,.2f}\n")
    print(f"   - Potential Profit/Loss: {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)\n")

    print(f"   - Portfolio Action Signal: {action_signal}")
    print(f"   - Justification: {justification}")
    print("---" * 20)


def print_investment_profile_analysis(df: pd.DataFrame, forecast: pd.DataFrame, signal: str, reason: str,
                                      risk_score: str, yield_status: str, projected_yield: float):
    """
    Provides an in-depth analysis for different investment styles if no portfolio is provided.
    """
    # --- Trend Analysis for Long-Term Outlook ---
    # Compare the trend at the end of the historical data vs. the end of the forecast
    len_df = len(df)
    trend_start = forecast.iloc[len_df - 1]['trend']
    trend_end = forecast.iloc[-1]['trend']

    if trend_end > trend_start:
        long_term_outlook = "The model identifies a positive underlying growth trend, suggesting potential for long-term appreciation."
    elif trend_end < trend_start:
        long_term_outlook = "The model identifies a negative underlying trend, suggesting caution for long-term holding."
    else:
        long_term_outlook = "The model's identified trend is flat, suggesting stability but limited long-term growth."
    long_term_outlook += " (Note: This is based on the model's trend component, not a guarantee)."

    # --- Dividend Analysis ---
    if yield_status == "Good":
        dividend_outlook = f"Excellent. The projected yield of {projected_yield:.2f}% is strong."
        if "Low" in risk_score or "Very Low" in risk_score:
            dividend_outlook += " Combined with a low risk score, this stock appears to be a solid candidate for dividend-focused investors."
        else:
            dividend_outlook += f" However, the {risk_score} risk level suggests investors should weigh the high yield against potential price volatility."
    elif yield_status == "Average":
        dividend_outlook = f"Average. The projected yield of {projected_yield:.2f}% provides some income, but may not be compelling for dedicated dividend investors."
    else:  # Low
        dividend_outlook = f"Low. With a projected yield of {projected_yield:.2f}%, this stock is not ideal for an income-focused strategy."

    print("\n" + "---" * 20)
    print("🧐           INVESTMENT PROFILE ANALYSIS")
    print("---" * 20)

    print("\n🔹 Short-Term Gain (Trader's Outlook):")
    print(f"   - Signal: {signal}")
    print(f"   - Outlook: {reason} The risk score is {risk_score}, which should be considered.")

    print("\n🔹 Long-Term Gain (Investor's Outlook):")
    print(f"   - Outlook: {long_term_outlook}")

    print("\n🔹 Dividend Income (Income Investor's Outlook):")
    print(f"   - Outlook: {dividend_outlook}")
    print("---" * 20)


def main():
    """
    Main function to parse arguments and run the analysis workflow.
    """
    # Using argparse for a more professional and user-friendly CLI
    parser = argparse.ArgumentParser(
        description="Prophet-based Stock Analysis Tool.",
        formatter_class=argparse.RawTextHelpFormatter  # For better help text formatting
    )
    parser.add_argument("ticker", help="The stock ticker symbol to analyze (e.g., 'AAPL', 'GOOGL').")
    parser.add_argument("--buy", type=float, help="Optional: The price at which you bought the stock.")
    parser.add_argument("--units", type=int, help="Optional: The number of units you own.")
    parser.add_argument('--no-plots', action='store_false', dest='show_plots',
                        help="Prevent forecast plots from being displayed.")
    parser.set_defaults(show_plots=SHOW_PLOTS)

    args = parser.parse_args()

    # Validate that if one portfolio argument is given, the other must be too.
    if (args.buy is not None and args.units is None) or \
            (args.buy is None and args.units is not None):
        parser.error("Arguments --buy and --units must be used together.")

    # --- Run Analysis ---
    stock_df, stock_name, historical_dividends, baseline_dividend, currency, fifty_two_week_high, fifty_two_week_low = get_stock_data(
        args.ticker, START_DATE_STR)

    if stock_df is not None and not stock_df.empty:
        display_name = f"{stock_name} ({args.ticker.upper()})"

        forecast_data, signal, reason, risk_score, yield_status, projected_yield = analyze_and_forecast(
            stock_df, display_name, FORECAST_DAYS, baseline_dividend, historical_dividends, show_plots=args.show_plots,
            currency=currency, fifty_two_week_high=fifty_two_week_high, fifty_two_week_low=fifty_two_week_low
        )

        # If portfolio details were provided, run the portfolio analysis
        if args.buy is not None and args.units is not None and forecast_data is not None:
            display_portfolio_analysis(forecast_data, args.buy, args.units, signal, reason, currency)
        # Otherwise, run the general investment profile analysis
        elif forecast_data is not None:
            print_investment_profile_analysis(
                stock_df, forecast_data, signal, reason, risk_score, yield_status, projected_yield
            )


if __name__ == "__main__":
    main()
