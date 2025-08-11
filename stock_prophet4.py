# -*- coding: utf-8 -*-
"""
Enhanced Prophet-Based Stock Analysis Tool (Merged Version)

This script combines the original Prophet-based analysis with enhanced analytics:
- Retains all original output formats and analysis from stock_prophet2.py
- Adds enhanced technical indicators and market sentiment analysis
- Provides both detailed technical analysis and simplified explanations
- Includes accuracy testing and confidence scoring
- Maintains backward compatibility with original CLI interface

Features from original (stock_prophet2.py):
- Professional Prophet forecasting with data scaling
- Comprehensive dividend analysis and portfolio projections
- Risk assessment and investment profile analysis
- U.S. holidays integration and robust data fetching

Enhanced features from stock_prophet3.py:
- Technical indicators (moving averages, momentum, volume analysis)
- Market sentiment analysis (VIX fear index)
- Prediction accuracy testing on historical data
- Enhanced confidence scoring and signal analysis
- Simplified explanations alongside technical details
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from prophet import Prophet
import warnings
import argparse
from typing import Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Suppress informational messages from Prophet and runtime warnings from numpy
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# #################################
# ##     CONFIGURATION & THRESHOLDS   ##
# #################################

# Historical data settings - extended for better pattern recognition
START_DATE_STR = '2020-01-01'  # Extended from 2023 for better patterns (4+ years)
FORECAST_DAYS = 90

# Analysis thresholds (from original)
YIELD_THRESHOLD_GOOD = 4.0
YIELD_THRESHOLD_AVG = 2.0
SIGNAL_PRICE_CHANGE_PCT = 5.0
RISK_METRIC_LOW = 5.0
RISK_METRIC_MEDIUM = 15.0
RISK_METRIC_HIGH = 25.0

# Enhanced analysis thresholds
TECHNICAL_SIGNAL_THRESHOLD = 3  # Minimum positive signals for strong buy
HIGH_CONFIDENCE_UNCERTAINTY = 15.0  # Max uncertainty % for high confidence
MEDIUM_CONFIDENCE_UNCERTAINTY = 25.0  # Max uncertainty % for medium confidence

# Display and output configuration
SHOW_PLOTS = False  # Set to False to disable plots by default
# Options: True (show plots), False (no plots)
# Can be overridden with --no-plots CLI flag

SHOW_SIMPLE = False  # Set to True to show simplified explanations by default


# Options: True (show simple explanations), False (technical only)
# Can be overridden with --simple CLI flag


# #############################
# ##     ENHANCED DATA FETCHING     ##
# #############################

def get_stock_data_enhanced(ticker: str, start_date_str: str) -> Tuple[
    Optional[pd.DataFrame], str, Optional[pd.Series], float, str, float, float, Dict]:
    """
    Enhanced version of original get_stock_data with market sentiment data.
    Returns all original data plus market indicators.
    """
    try:
        # Use the yfinance Ticker object (original logic maintained)
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        stock_name = stock_info.get('longName', ticker)
        currency = stock_info.get('currency', '

        # #############################
        # ##     ENHANCED TECHNICAL ANALYSIS     ##
        # #############################


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for enhanced analysis.
    Adds indicators without modifying original data structure.
    """
    result = df.copy()

    try:
        # Moving averages for trend analysis
        result['sma_20'] = df['y'].rolling(window=20).mean()
        result['sma_50'] = df['y'].rolling(window=50).mean()
        result['sma_200'] = df['y'].rolling(window=200).mean()

        # Daily returns and volatility
        result['daily_return'] = df['y'].pct_change()
        result['volatility_20'] = result['daily_return'].rolling(window=20).std() * np.sqrt(252)

        # Volume analysis (if available)
        if 'Volume' in df.columns:
            result['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = df['Volume'] / result['volume_sma_20']
            result['volume_ratio'] = result['volume_ratio'].fillna(1.0)

        # Momentum indicators
        result['price_momentum_10'] = df['y'].pct_change(periods=10) * 100
        result['price_momentum_20'] = df['y'].pct_change(periods=20) * 100

        # Price position in trading range
        result['high_20'] = df['High'].rolling(window=20).max() if 'High' in df.columns else df['y'].rolling(
            window=20).max()
        result['low_20'] = df['Low'].rolling(window=20).min() if 'Low' in df.columns else df['y'].rolling(
            window=20).min()

        price_range = result['high_20'] - result['low_20']
        result['price_position'] = (df['y'] - result['low_20']) / price_range
        result['price_position'] = result['price_position'].fillna(0.5)

        # Fill NaN values with reasonable defaults
        result = result.fillna(method='bfill').fillna(method='ffill')

    except Exception as e:
        print(f"⚠️ Warning: Could not calculate some technical indicators: {e}")

    return result


def analyze_market_sentiment(market_data: Dict, current_date: pd.Timestamp = None) -> Dict:
    """
    Analyze market sentiment using VIX and other indicators.
    """
    sentiment = {}

    try:
        # VIX Analysis (Fear Index)
        if 'vix' in market_data and not market_data['vix'].empty:
            recent_vix = market_data['vix'].tail(10).mean()
            current_vix = market_data['vix'].iloc[-1]

            if current_vix > 30:
                sentiment['fear_level'] = "High Fear"
                sentiment['fear_explanation'] = "Market shows high fear/volatility - may impact all stocks negatively"
                sentiment['fear_score'] = -2
            elif current_vix > 20:
                sentiment['fear_level'] = "Moderate Fear"
                sentiment['fear_explanation'] = "Market shows some concern - normal market conditions"
                sentiment['fear_score'] = -1
            else:
                sentiment['fear_level'] = "Low Fear"
                sentiment['fear_explanation'] = "Market confidence is high - good environment for stocks"
                sentiment['fear_score'] = 1

            sentiment['vix_value'] = current_vix

        # Market trend analysis (if SPY data available)
        if 'spy' in market_data and not market_data['spy'].empty:
            spy_recent = market_data['spy'].tail(20)
            spy_change = (spy_recent.iloc[-1] - spy_recent.iloc[0]) / spy_recent.iloc[0] * 100

            if spy_change > 2:
                sentiment['market_trend'] = "Bullish"
                sentiment['market_trend_score'] = 1
            elif spy_change < -2:
                sentiment['market_trend'] = "Bearish"
                sentiment['market_trend_score'] = -1
            else:
                sentiment['market_trend'] = "Neutral"
                sentiment['market_trend_score'] = 0

    except Exception as e:
        print(f"⚠️ Warning: Could not analyze market sentiment: {e}")

    return sentiment


# #############################
# ##     ENHANCED FORECASTING     ##
# #############################

def analyze_and_forecast_enhanced(df: pd.DataFrame, display_name: str, forecast_days: int,
                                  baseline_dividend: float, historical_dividends: pd.Series,
                                  show_plots: bool, currency: str, fifty_two_week_high: float,
                                  fifty_two_week_low: float, market_data: Dict) -> Tuple[
    Optional[pd.DataFrame], str, str, str, str, float, Dict]:
    """
    Enhanced version of original analyze_and_forecast function.
    Maintains all original functionality while adding enhanced analysis.
    """
    # Calculate technical indicators first
    enhanced_df = calculate_technical_indicators(df)

    # ORIGINAL PROPHET LOGIC (maintained exactly)
    # --- Data Transformation & Scaling ---
    df['y_log'] = np.log(df['y'])
    y_log_min = df['y_log'].min()
    y_log_max = df['y_log'].max()
    df['y_scaled'] = (df['y_log'] - y_log_min) / (y_log_max - y_log_min)

    # --- Model Training (original) ---
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_country_holidays(country_name='US')
    model.fit(df[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'}))

    # --- Forecasting (original) ---
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # --- Inverse Transform & Analysis (original) ---
    for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
        if col in forecast.columns:
            forecast[col] = forecast[col] * (y_log_max - y_log_min) + y_log_min

    current_price = df['y'].iloc[-1]
    future_prediction = forecast.iloc[-1]
    predicted_price = np.exp(future_prediction['yhat'])
    predicted_lower = np.exp(future_prediction['yhat_lower'])
    predicted_upper = np.exp(future_prediction['yhat_upper'])

    # ENHANCED ANALYSIS: Technical signals
    enhanced_analysis = analyze_technical_signals(enhanced_df, market_data, predicted_price, current_price)

    # ENHANCED ANALYSIS: Prediction confidence testing
    confidence_metrics = test_prediction_confidence(enhanced_df, forecast_days)

    # ORIGINAL ANALYSIS REPORT (maintained)
    signal, reason, risk_score, yield_status, projected_yield = print_analysis_report_enhanced(
        df, enhanced_df, forecast, display_name, current_price, predicted_price,
        predicted_lower, predicted_upper, baseline_dividend, historical_dividends,
        currency, fifty_two_week_high, fifty_two_week_low, enhanced_analysis, confidence_metrics
    )

    # ORIGINAL VISUALIZATION (maintained but enhanced)
    if show_plots:
        print("\n📈 Generating enhanced forecast plots...")
        try:
            create_enhanced_plots(df, enhanced_df, forecast, display_name, currency, market_data)
        except Exception as e:
            print(f"Could not generate plots. Error: {e}")

    # Return original format plus enhanced data
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

    return forecast, signal, reason, risk_score, yield_status, projected_yield, enhanced_analysis


def analyze_technical_signals(enhanced_df: pd.DataFrame, market_data: Dict,
                              predicted_price: float, current_price: float) -> Dict:
    """
    Analyze technical indicators to generate enhanced signals.
    """
    signals = {'positive': 0, 'negative': 0, 'details': []}

    try:
        latest = enhanced_df.iloc[-1]

        # Moving average signals
        if not pd.isna(latest.get('sma_20')) and not pd.isna(latest.get('sma_50')):
            if current_price > latest['sma_20'] > latest['sma_50']:
                signals['positive'] += 2
                signals['details'].append("Strong uptrend: Price above both 20-day and 50-day averages")
            elif current_price > latest['sma_20']:
                signals['positive'] += 1
                signals['details'].append("Mild uptrend: Price above 20-day average")
            elif current_price < latest['sma_20'] < latest['sma_50']:
                signals['negative'] += 2
                signals['details'].append("Strong downtrend: Price below both 20-day and 50-day averages")
            elif current_price < latest['sma_20']:
                signals['negative'] += 1
                signals['details'].append("Mild downtrend: Price below 20-day average")

        # Momentum signals
        if not pd.isna(latest.get('price_momentum_10')):
            momentum = latest['price_momentum_10']
            if momentum > 5:
                signals['positive'] += 1
                signals['details'].append(f"Strong positive momentum: +{momentum:.1f}% over 10 days")
            elif momentum < -5:
                signals['negative'] += 1
                signals['details'].append(f"Strong negative momentum: {momentum:.1f}% over 10 days")

        # Volume analysis
        if not pd.isna(latest.get('volume_ratio')):
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 1.5:
                signals['positive'] += 1
                signals['details'].append(f"High trading activity: {vol_ratio:.1f}x normal volume")
            elif vol_ratio < 0.5:
                signals['negative'] += 1
                signals['details'].append(f"Low trading activity: {vol_ratio:.1f}x normal volume")

        # Price position in range
        if not pd.isna(latest.get('price_position')):
            position = latest['price_position']
            if position > 0.8:
                signals['positive'] += 1
                signals['details'].append("Price near 20-day high - strong momentum")
            elif position < 0.2:
                signals['negative'] += 1
                signals['details'].append("Price near 20-day low - weak momentum")

        # Market sentiment integration
        market_sentiment = analyze_market_sentiment(market_data)
        if market_sentiment.get('fear_score'):
            fear_score = market_sentiment['fear_score']
            if fear_score > 0:
                signals['positive'] += fear_score
                signals['details'].append(f"Market sentiment: {market_sentiment['fear_level']}")
            else:
                signals['negative'] += abs(fear_score)
                signals['details'].append(f"Market sentiment: {market_sentiment['fear_level']}")

        # Prediction magnitude
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        if abs(price_change_pct) > 10:
            if price_change_pct > 0:
                signals['positive'] += 2
                signals['details'].append(f"Strong predicted growth: +{price_change_pct:.1f}%")
            else:
                signals['negative'] += 2
                signals['details'].append(f"Strong predicted decline: {price_change_pct:.1f}%")
        elif abs(price_change_pct) > 5:
            if price_change_pct > 0:
                signals['positive'] += 1
                signals['details'].append(f"Moderate predicted growth: +{price_change_pct:.1f}%")
            else:
                signals['negative'] += 1
                signals['details'].append(f"Moderate predicted decline: {price_change_pct:.1f}%")

    except Exception as e:
        print(f"⚠️ Warning: Could not complete technical analysis: {e}")

    return signals


def test_prediction_confidence(df: pd.DataFrame, forecast_days: int) -> Dict:
    """
    Test prediction accuracy on historical data to build confidence metrics.
    """
    confidence = {'tested': False, 'accuracy': 'Unknown', 'details': {}}

    try:
        if len(df) < 150:  # Need sufficient data for testing
            return confidence

        # Test on 2 different historical periods
        test_periods = [30, 60]
        all_errors = []

        for days in test_periods:
            for offset in [60, 120]:  # Test at different time points
                if len(df) <= days + offset:
                    continue

                # Split data
                train_end = len(df) - offset - days
                if train_end < 60:
                    continue

                train_data = df.iloc[:train_end].copy()
                actual_data = df.iloc[train_end:train_end + days].copy()

                if len(actual_data) < days:
                    continue

                try:
                    # Create mini-forecast
                    test_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                    test_model.fit(train_data[['ds', 'y']])

                    test_future = test_model.make_future_dataframe(periods=days)
                    test_forecast = test_model.predict(test_future)

                    # Compare predictions vs actual
                    predicted = test_forecast.tail(days)['yhat'].values
                    actual = actual_data['y'].values

                    if len(predicted) == len(actual):
                        errors = np.abs(predicted - actual) / actual * 100
                        all_errors.extend(errors)

                except:
                    continue

        if all_errors:
            avg_error = np.mean(all_errors)
            confidence['tested'] = True
            confidence['average_error_percent'] = avg_error
            confidence['tests_performed'] = len(all_errors)

            if avg_error < 5:
                confidence['accuracy'] = 'High'
            elif avg_error < 15:
                confidence['accuracy'] = 'Medium'
            else:
                confidence['accuracy'] = 'Low'

            confidence['details'] = {
                'avg_error_pct': avg_error,
                'error_range': [np.min(all_errors), np.max(all_errors)]
            }

    except Exception as e:
        print(f"⚠️ Warning: Could not test prediction confidence: {e}")

    return confidence


# #############################
# ##     ENHANCED REPORTING     ##
# #############################

def print_analysis_report_enhanced(df, enhanced_df, forecast, display_name, current_price, predicted_price,
                                   predicted_lower, predicted_upper, baseline_dividend, historical_dividends,
                                   currency, fifty_two_week_high, fifty_two_week_low, enhanced_analysis,
                                   confidence_metrics) -> Tuple[str, str, str, str, float]:
    """
    Enhanced version of original print_analysis_report.
    Maintains all original output while adding enhanced insights.
    """
    # ORIGINAL REPORT LOGIC (maintained exactly)
    last_known_date_obj = df['ds'].iloc[-1]
    predicted_date_obj = forecast['ds'].iloc[-1]

    # Original dividend analysis
    projected_yield = (baseline_dividend / predicted_price) * 100 if predicted_price > 0 else 0
    if projected_yield > YIELD_THRESHOLD_GOOD:
        yield_status = "Good"
    elif projected_yield > YIELD_THRESHOLD_AVG:
        yield_status = "Average"
    else:
        yield_status = "Low"

    # Original signal logic
    price_change_pct = ((predicted_price - current_price) / current_price) * 100
    if price_change_pct > SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📈 BUY", f"Price is forecast to rise by {price_change_pct:.2f}%."
    elif price_change_pct < -SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📉 SELL", f"Price is forecast to fall by {price_change_pct:.2f}%."
    else:
        signal, reason = "🤔 HOLD", f"Price is forecast to be stable (change: {price_change_pct:.2f}%)."

    # Original risk assessment
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

    # ENHANCED SIGNAL GENERATION
    enhanced_signal, enhanced_reason, enhanced_confidence = generate_enhanced_recommendation(
        signal, reason, enhanced_analysis, confidence_metrics, risk_metric
    )

    # ORIGINAL HEADER AND OUTPUT (maintained exactly)
    analysis_time = datetime.now(ZoneInfo("Asia/Tokyo")).strftime('%d-%b-%Y %I:%M:%S %p %Z')
    header = f"ENHANCED PROPHET-BASED STOCK ANALYSIS: {display_name}"
    print("\n" + "=" * 80)
    print(header.center(80))
    print(f"Analysis Time: {analysis_time}".center(80))
    print("=" * 80)
    print(f"Forecasting {FORECAST_DAYS} days (until {predicted_date_obj.strftime('%d-%b-%Y')}).\n")

    print(f"🔹 Last Market Data (as of {last_known_date_obj.strftime('%d-%b-%Y')}):")
    print(f"   - Closing Price: {currency} {current_price:,.2f}")

    # ORIGINAL Historical High/Low Information (maintained)
    print(f"   - Historical High / Low:")
    df_by_year = df.groupby(df['ds'].dt.year)

    for year, year_df in df_by_year:
        high_col = 'High' if 'High' in year_df.columns else 'y'
        low_col = 'Low' if 'Low' in year_df.columns else 'y'

        high_row = year_df.loc[year_df[high_col].idxmax()]
        year_high_price = high_row[high_col]
        year_high_date = high_row['ds'].strftime('%d-%b-%Y')

        low_row = year_df.loc[year_df[low_col].idxmin()]
        year_low_price = low_row[low_col]
        year_low_date = low_row['ds'].strftime('%d-%b-%Y')

        print(
            f"     - {year} High / Low: {currency} {year_high_price:,.2f} ({year_high_date}) / {currency} {year_low_price:,.2f} ({year_low_date})")

    print(f"\n   - 52-Week High / Low: {currency} {fifty_two_week_high:,.2f} / {currency} {fifty_two_week_low:,.2f}\n")

    # ORIGINAL Forecast section (maintained)
    print(f"🔮 Forecast for {predicted_date_obj.strftime('%d-%b-%Y')}:")
    print(f"   - Predicted Price: {currency} {predicted_price:,.2f}")
    print(f"   - Expected Range: {currency} {predicted_lower:,.2f} to {currency} {predicted_upper:,.2f}\n")

    # ENHANCED: Technical Analysis Summary
    print(f"📊 Technical Analysis Summary:")
    total_signals = enhanced_analysis['positive'] + enhanced_analysis['negative']
    net_signals = enhanced_analysis['positive'] - enhanced_analysis['negative']

    print(f"   - Positive Signals: {enhanced_analysis['positive']}")
    print(f"   - Negative Signals: {enhanced_analysis['negative']}")
    print(f"   - Net Signal Strength: {net_signals:+d}")

    if enhanced_analysis['details']:
        print(f"   - Key Technical Factors:")
        for detail in enhanced_analysis['details'][:3]:  # Show top 3 factors
            print(f"     • {detail}")
    print()

    # ENHANCED: Confidence Analysis
    if confidence_metrics['tested']:
        print(f"🎯 Prediction Confidence Analysis:")
        print(f"   - Historical Accuracy: {confidence_metrics['accuracy']}")
        print(f"   - Average Prediction Error: {confidence_metrics['average_error_percent']:.1f}%")
        print(f"   - Based on {confidence_metrics['tests_performed']} historical tests")
        if confidence_metrics['accuracy'] == 'High':
            print(f"   - Confidence Level: Our model has been quite accurate on similar predictions")
        elif confidence_metrics['accuracy'] == 'Medium':
            print(f"   - Confidence Level: Our model shows reasonable accuracy with some variation")
        else:
            print(f"   - Confidence Level: Predictions have higher uncertainty - use with caution")
        print()

    # ORIGINAL Dividend Analysis (maintained exactly)
    print(f"💰 Dividend Yield Analysis:")
    print(
        f"   - Projected Yield: {projected_yield:.2f}% ({currency} {baseline_dividend:.4f} per share) (based on a predicted price of {currency} {predicted_price:,.2f})")
    print(f"   - Projection Status: {yield_status}")

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
            print()
    else:
        print("   - Historical Payout: No dividends were paid in the tracked period.\n")

    # ENHANCED Buy/Sell Signal (combines original + enhanced)
    print(f"📊 Investment Recommendation:")
    print(f"   - Primary Signal: {enhanced_signal}")
    print(f"   - Enhanced Analysis: {enhanced_reason}")
    print(f"   - Original Prophet Signal: {signal} - {reason}")
    print(f"   - Confidence: {enhanced_confidence}\n")

    # ORIGINAL Risk Assessment (maintained)
    print(f"⚠️ Risk Assessment:")
    print(f"   - Model Uncertainty ({risk_metric:.2f}%) suggests the risk is: {risk_score}")

    # ENHANCED: Additional risk context
    if risk_metric > 30:
        print(f"   - High uncertainty suggests careful position sizing and close monitoring")
    elif risk_metric < 10:
        print(f"   - Low uncertainty suggests relatively stable prediction")

    print("=" * 80)

    # Return original format
    return enhanced_signal, enhanced_reason, risk_score, yield_status, projected_yield


def generate_enhanced_recommendation(original_signal: str, original_reason: str,
                                     enhanced_analysis: Dict, confidence_metrics: Dict, risk_metric: float) -> Tuple[
    str, str, str]:
    """
    Generate enhanced recommendation combining original Prophet analysis with technical analysis.
    """
    positive_signals = enhanced_analysis['positive']
    negative_signals = enhanced_analysis['negative']
    net_signals = positive_signals - negative_signals

    # Determine confidence level
    if confidence_metrics.get('tested', False):
        accuracy = confidence_metrics['accuracy']
        avg_error = confidence_metrics.get('average_error_percent', 20)

        if accuracy == 'High' and risk_metric < HIGH_CONFIDENCE_UNCERTAINTY:
            confidence = "High"
        elif accuracy in ['High', 'Medium'] and risk_metric < MEDIUM_CONFIDENCE_UNCERTAINTY:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        if risk_metric < HIGH_CONFIDENCE_UNCERTAINTY:
            confidence = "Medium"
        else:
            confidence = "Low"

    # Enhanced signal generation combining technical and Prophet analysis
    if net_signals >= TECHNICAL_SIGNAL_THRESHOLD and "BUY" in original_signal:
        enhanced_signal = "🚀 STRONG BUY"
        enhanced_reason = f"Both technical analysis ({positive_signals} positive vs {negative_signals} negative signals) and Prophet forecast strongly support buying"
    elif net_signals >= 1 and "BUY" in original_signal:
        enhanced_signal = "📈 BUY"
        enhanced_reason = f"Technical indicators ({positive_signals} positive signals) support the Prophet buy signal"
    elif net_signals <= -TECHNICAL_SIGNAL_THRESHOLD and "SELL" in original_signal:
        enhanced_signal = "🔻 STRONG SELL"
        enhanced_reason = f"Both technical analysis ({negative_signals} negative vs {positive_signals} positive signals) and Prophet forecast strongly support selling"
    elif net_signals <= -1 and "SELL" in original_signal:
        enhanced_signal = "📉 SELL"
        enhanced_reason = f"Technical indicators ({negative_signals} negative signals) support the Prophet sell signal"
    elif net_signals >= 2:
        enhanced_signal = "📈 TECHNICAL BUY"
        enhanced_reason = f"Strong technical signals ({positive_signals} positive vs {negative_signals} negative) suggest buying despite Prophet neutrality"
    elif net_signals <= -2:
        enhanced_signal = "📉 TECHNICAL SELL"
        enhanced_reason = f"Strong negative technical signals ({negative_signals} negative vs {positive_signals} positive) suggest selling despite Prophet neutrality"
    elif "BUY" in original_signal and net_signals >= 0:
        enhanced_signal = "📈 CAUTIOUS BUY"
        enhanced_reason = f"Prophet suggests buying but technical signals are mixed ({positive_signals} positive, {negative_signals} negative)"
    elif "SELL" in original_signal and net_signals <= 0:
        enhanced_signal = "📉 CAUTIOUS SELL"
        enhanced_reason = f"Prophet suggests selling and technical signals don't contradict ({positive_signals} positive, {negative_signals} negative)"
    else:
        enhanced_signal = "🤔 HOLD"
        enhanced_reason = f"Mixed signals from Prophet and technical analysis - wait for clearer direction"

    return enhanced_signal, enhanced_reason, confidence


# #############################
# ##     ENHANCED VISUALIZATION     ##
# #############################

def create_enhanced_plots(df: pd.DataFrame, enhanced_df: pd.DataFrame, forecast: pd.DataFrame,
                          display_name: str, currency: str, market_data: Dict):
    """
    Create enhanced plots combining original Prophet visualization with technical analysis.
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Original Prophet forecast with technical overlays
        ax1.plot(df['ds'], df['y'], 'k.', label='Historical Price', alpha=0.7)
        ax1.plot(forecast['ds'], np.exp(forecast['yhat']), ls='-', c='#0072B2', linewidth=2, label='Prophet Forecast')
        ax1.fill_between(forecast['ds'], np.exp(forecast['yhat_lower']), np.exp(forecast['yhat_upper']),
                         color='#0072B2', alpha=0.2, label='80% Confidence Interval')

        # Add technical indicators if available
        if 'sma_20' in enhanced_df.columns:
            ax1.plot(enhanced_df['ds'], enhanced_df['sma_20'], '--', color='orange', alpha=0.8, label='20-day SMA')
        if 'sma_50' in enhanced_df.columns:
            ax1.plot(enhanced_df['ds'], enhanced_df['sma_50'], '--', color='red', alpha=0.8, label='50-day SMA')

        ax1.set_title(f'Enhanced Price Forecast: {display_name}', size=14, fontweight='bold')
        ax1.set_ylabel(f"Price ({currency})", size=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volume analysis
        if 'Volume' in enhanced_df.columns:
            ax2.bar(enhanced_df['ds'], enhanced_df['Volume'], alpha=0.6, color='lightblue', width=1)
            if 'volume_sma_20' in enhanced_df.columns:
                ax2.plot(enhanced_df['ds'], enhanced_df['volume_sma_20'], 'red', linewidth=2, label='20-day Avg Volume')
            ax2.set_title('Trading Volume Analysis', size=14, fontweight='bold')
            ax2.set_ylabel('Volume', size=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Volume Analysis (N/A)', size=14)

        # Plot 3: Technical momentum indicators
        if 'price_momentum_10' in enhanced_df.columns:
            ax3.plot(enhanced_df['ds'], enhanced_df['price_momentum_10'], color='green', alpha=0.7,
                     label='10-day Momentum')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax3.axhline(y=-5, color='blue', linestyle='--', alpha=0.5, label='Oversold')
            ax3.fill_between(enhanced_df['ds'], 0, enhanced_df['price_momentum_10'],
                             where=(enhanced_df['price_momentum_10'] > 0), color='green', alpha=0.3)
            ax3.fill_between(enhanced_df['ds'], 0, enhanced_df['price_momentum_10'],
                             where=(enhanced_df['price_momentum_10'] < 0), color='red', alpha=0.3)
            ax3.set_title('Price Momentum (10-day % change)', size=14, fontweight='bold')
            ax3.set_ylabel('Momentum %', size=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Momentum data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Momentum Analysis (N/A)', size=14)

        # Plot 4: Market sentiment (VIX) if available
        if 'vix' in market_data and not market_data['vix'].empty:
            vix_data = market_data['vix']
            ax4.plot(vix_data.index, vix_data.values, color='purple', linewidth=2, label='VIX (Fear Index)')
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Low Fear')
            ax4.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='High Fear')
            ax4.fill_between(vix_data.index, 0, vix_data.values,
                             where=(vix_data.values > 30), color='red', alpha=0.3, label='Fear Zone')
            ax4.fill_between(vix_data.index, 0, vix_data.values,
                             where=(vix_data.values < 20), color='green', alpha=0.3, label='Confidence Zone')
            ax4.set_title('Market Sentiment (VIX Fear Index)', size=14, fontweight='bold')
            ax4.set_ylabel('VIX Level', size=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Market sentiment data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Market Sentiment (N/A)', size=14)

        plt.tight_layout()
        plt.show()

        # Create original Prophet components plot
        # Recreate model for components (since we need the fitted model)
        temp_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        temp_model.add_country_holidays(country_name='US')
        df_temp = df[['ds', 'y']].copy()
        df_temp['y'] = np.log(df_temp['y'])  # Use log transform like in main analysis
        temp_model.fit(df_temp)
        temp_future = temp_model.make_future_dataframe(periods=FORECAST_DAYS)
        temp_forecast = temp_model.predict(temp_future)

        temp_model.plot_components(temp_forecast)
        plt.suptitle('Prophet Model Components Analysis', size=16, fontweight='bold')
        plt.show()

        print("\n📊 CHART EXPLANATION:")
        print("   📈 Top-left: Price history, Prophet forecast, and moving averages")
        print("   📊 Top-right: Trading volume and average volume trends")
        print("   🔄 Bottom-left: Price momentum indicators (positive = upward pressure)")
        print("   😰 Bottom-right: Market fear index (VIX) - higher values mean more market anxiety")
        print("   📉 Components: Prophet's breakdown of trend, seasonality, and holiday effects")

    except Exception as e:
        # Fall back to original simple plot
        print(f"⚠️ Could not create enhanced plots, showing basic forecast: {e}")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['ds'], df['y'], 'k.', label='Historical Price')
        ax.plot(forecast['ds'], np.exp(forecast['yhat']), ls='-', c='#0072B2', label='Forecast')
        ax.fill_between(forecast['ds'], np.exp(forecast['yhat_lower']), np.exp(forecast['yhat_upper']),
                        color='#0072B2', alpha=0.2, label='80% Confidence Interval')
        ax.set_title(f'Price Forecast for {display_name}', size=16)
        ax.set_xlabel("Date", size=12)
        ax.set_ylabel(f"Price ({currency})", size=12)
        ax.legend()
        ax.grid(True, which='both', ls='--', lw=0.5)
        plt.show()


# #############################
# ##     ORIGINAL FUNCTIONS (MAINTAINED)     ##
# #############################

def display_portfolio_analysis(forecast: pd.DataFrame, buy_price: float, units: int, signal: str, reason: str,
                               currency: str):
    """
    Original portfolio analysis function - maintained exactly.
    """
    predicted_price = forecast.iloc[-1]['yhat']
    predicted_date = forecast.iloc[-1]['ds'].date().strftime('%d-%b-%Y')

    initial_cost = buy_price * units
    predicted_value = predicted_price * units
    potential_pl = predicted_value - initial_cost
    potential_pl_percent = (potential_pl / initial_cost) * 100 if initial_cost > 0 else 0

    if "BUY" in signal:
        action_signal = "📈 Accumulate More"
        justification = f"The forecast is positive ({reason}) and your position is projected to grow by {potential_pl_percent:+.2f}%. This could be a good opportunity to increase your holdings."
    elif "SELL" in signal:
        action_signal = "📉 Secure Profits / Cut Losses"
        if potential_pl > 0:
            justification = f"The forecast is negative ({reason}). Consider selling to secure your current profit of {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
        else:
            justification = f"The forecast is negative ({reason}). Consider selling to prevent further losses on your position, currently at {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
    else:
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
    Original investment profile analysis function - maintained exactly.
    """
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

    if yield_status == "Good":
        dividend_outlook = f"Excellent. The projected yield of {projected_yield:.2f}% is strong."
        if "Low" in risk_score or "Very Low" in risk_score:
            dividend_outlook += " Combined with a low risk score, this stock appears to be a solid candidate for dividend-focused investors."
        else:
            dividend_outlook += f" However, the {risk_score} risk level suggests investors should weigh the high yield against potential price volatility."
    elif yield_status == "Average":
        dividend_outlook = f"Average. The projected yield of {projected_yield:.2f}% provides some income, but may not be compelling for dedicated dividend investors."
    else:
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


# #############################
# ##     ENHANCED MAIN FUNCTION     ##
# #############################

def main():
    """
    Enhanced main function maintaining original CLI interface with enhanced functionality.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Prophet-based Stock Analysis Tool with Technical Analysis.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("ticker", help="The stock ticker symbol to analyze (e.g., 'AAPL', 'GOOGL').")
    parser.add_argument("--buy", type=float, help="Optional: The price at which you bought the stock.")
    parser.add_argument("--units", type=int, help="Optional: The number of units you own.")
    parser.add_argument('--no-plots', action='store_false', dest='show_plots',
                        help="Prevent forecast plots from being displayed.")
    parser.add_argument('--simple', action='store_true', dest='show_simple',
                        help="Show simplified analysis explanations alongside technical details.")
    parser.set_defaults(show_plots=SHOW_PLOTS, show_simple=SHOW_SIMPLE)

    args = parser.parse_args()

    if (args.buy is not None and args.units is None) or \
            (args.buy is None and args.units is not None):
        parser.error("Arguments --buy and --units must be used together.")

    print("🚀 Enhanced Prophet-Based Stock Analysis Tool")
    print("Combining time-series forecasting with technical analysis for better predictions\n")

    # Enhanced data fetching
    stock_df, stock_name, historical_dividends, baseline_dividend, currency, fifty_two_week_high, fifty_two_week_low, market_data = get_stock_data_enhanced(
        args.ticker, START_DATE_STR)

    if stock_df is not None and not stock_df.empty:
        display_name = f"{stock_name} ({args.ticker.upper()})"

        # Enhanced analysis
        forecast_data, signal, reason, risk_score, yield_status, projected_yield, enhanced_analysis = analyze_and_forecast_enhanced(
            stock_df, display_name, FORECAST_DAYS, baseline_dividend, historical_dividends,
            show_plots=args.show_plots, currency=currency, fifty_two_week_high=fifty_two_week_high,
            fifty_two_week_low=fifty_two_week_low, market_data=market_data
        )

        # Original portfolio or investment analysis
        if args.buy is not None and args.units is not None and forecast_data is not None:
            display_portfolio_analysis(forecast_data, args.buy, args.units, signal, reason, currency)

            # Enhanced: Additional portfolio insights
            if args.show_simple:
                print_simple_portfolio_explanation(forecast_data, args.buy, args.units, enhanced_analysis, currency)

        elif forecast_data is not None:
            print_investment_profile_analysis(
                stock_df, forecast_data, signal, reason, risk_score, yield_status, projected_yield
            )

            # Enhanced: Simple explanations if requested
            if args.show_simple:
                print_simple_investment_explanation(enhanced_analysis, signal, reason)


def print_simple_portfolio_explanation(forecast: pd.DataFrame, buy_price: float, units: int,
                                       enhanced_analysis: Dict, currency: str):
    """
    Provide simple explanations for portfolio analysis.
    """
    predicted_price = forecast.iloc[-1]['yhat']
    gain_loss = (predicted_price - buy_price) / buy_price * 100

    print(f"\n💡 SIMPLE EXPLANATION FOR YOUR INVESTMENT:")

    if gain_loss > 10:
        print(f"   🎉 Great news! Your investment might grow by about {gain_loss:.1f}%")
    elif gain_loss > 0:
        print(f"   📈 Good news! Small gains expected around {gain_loss:.1f}%")
    elif gain_loss > -10:
        print(f"   ⚠️  Small loss possible around {abs(gain_loss):.1f}% - monitor closely")
    else:
        print(f"   🔴 Significant loss possible around {abs(gain_loss):.1f}% - consider your options")

    print(
        f"   📊 This is based on {enhanced_analysis['positive']} positive and {enhanced_analysis['negative']} negative market signals")
    print(f"   📝 Remember: These are predictions based on patterns, not guarantees")


def print_simple_investment_explanation(enhanced_analysis: Dict, signal: str, reason: str):
    """
    Provide simple explanations for general investment analysis.
    """
    print(f"\n💡 SIMPLE EXPLANATION:")

    net_signals = enhanced_analysis['positive'] - enhanced_analysis['negative']

    if net_signals > 2:
        print(f"   🟢 Multiple positive signs suggest this might be a good buying opportunity")
    elif net_signals > 0:
        print(f"   🟡 Some positive signs, but be cautious - mixed signals present")
    elif net_signals < -2:
        print(f"   🔴 Multiple warning signs suggest avoiding or selling this stock")
    else:
        print(f"   🤔 Very mixed signals - probably best to wait and see what happens")

    print(f"   📈 Our computer model says: {reason}")
    print(
        f"   🔍 Market analysis found: {enhanced_analysis['positive']} positive and {enhanced_analysis['negative']} negative signals")

    if enhanced_analysis['details']:
        print(f"   📋 Main factors:")
        for i, detail in enumerate(enhanced_analysis['details'][:2], 1):
            print(f"      {i}. {detail}")


if __name__ == "__main__":
    main())

    # Get the trailing annual dividend rate for projections (original logic)
    baseline_dividend = stock_info.get('trailingAnnualDividendRate', 0.0)

    # Fetch history (original logic maintained)
    df = stock.history(start=start_date_str, auto_adjust=True)
    dividends = stock.dividends[start_date_str:]

    if df.empty:
        print(f"⚠️ No data found for '{ticker}' for the specified period.")
    return None, ticker, None, 0.0, currency, 0.0, 0.0, {}

    # Calculate ACCURATE 52-week high/low from our actual data (last 252 trading days)
    current_date = df.index[-1]
    one_year_ago = current_date - pd.Timedelta(days=365)
    last_52_weeks = df[df.index >= one_year_ago]

    if len(last_52_weeks) > 0:
        # Use actual historical data for accurate 52-week high/low
        if 'High' in last_52_weeks.columns and 'Low' in last_52_weeks.columns:
            fifty_two_week_high = last_52_weeks['High'].max()
            fifty_two_week_low = last_52_weeks['Low'].min()
        else:
            # Fallback to Close prices if High/Low not available
            fifty_two_week_high = last_52_weeks['Close'].max()
            fifty_two_week_low = last_52_weeks['Close'].min()
    else:
        # Fallback to overall data range if less than 1 year of data
        if 'High' in df.columns and 'Low' in df.columns:
            fifty_two_week_high = df['High'].max()
            fifty_two_week_low = df['Low'].min()
        else:
            fifty_two_week_high = df['Close'].max()
            fifty_two_week_low = df['Close'].min()

    print(
        f"📊 Calculated 52-week high/low from actual trading data: {currency} {fifty_two_week_high:.2f} / {currency} {fifty_two_week_low:.2f}")

    # Original data preparation
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df['ds'] = df['ds'].dt.tz_localize(None)
    df.dropna(subset=['y'], inplace=True)

    if (df['y'] <= 0).any():
        print(f"⚠️ Warning: Stock data for {ticker} contains zero/negative prices. Removing them.")
        df = df[df['y'] > 0]

    # ENHANCED: Fetch market sentiment data
    market_data = {}
    try:
        print("📊 Fetching market sentiment data...")
        # VIX (Fear Index)
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(start=start_date_str, auto_adjust=True)
        if not vix_data.empty:
            market_data['vix'] = vix_data['Close']

        # SPY for market comparison (optional)
        spy = yf.Ticker("SPY")
        spy_data = spy.history(start=start_date_str, auto_adjust=True)
        if not spy_data.empty:
            market_data['spy'] = spy_data['Close']

    except Exception as e:
        print(f"⚠️ Could not fetch market data: {e}")
        market_data = {}

    # Original success message
    start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
    formatted_start_date = start_date_obj.strftime('%d-%b-%Y')
    print(f"✅ Successfully fetched {len(df)} days of data (from {formatted_start_date}) for {stock_name} ({ticker}).")

    return df, stock_name, dividends, baseline_dividend, currency, fifty_two_week_high, fifty_two_week_low, market_data

except Exception as e:
print(f"❌ An unexpected error occurred while fetching data for '{ticker}': {e}")
return None, ticker, None, 0.0, '


# #############################
# ##     ENHANCED TECHNICAL ANALYSIS     ##
# #############################

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for enhanced analysis.
    Adds indicators without modifying original data structure.
    """
    result = df.copy()

    try:
        # Moving averages for trend analysis
        result['sma_20'] = df['y'].rolling(window=20).mean()
        result['sma_50'] = df['y'].rolling(window=50).mean()
        result['sma_200'] = df['y'].rolling(window=200).mean()

        # Daily returns and volatility
        result['daily_return'] = df['y'].pct_change()
        result['volatility_20'] = result['daily_return'].rolling(window=20).std() * np.sqrt(252)

        # Volume analysis (if available)
        if 'Volume' in df.columns:
            result['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = df['Volume'] / result['volume_sma_20']
            result['volume_ratio'] = result['volume_ratio'].fillna(1.0)

        # Momentum indicators
        result['price_momentum_10'] = df['y'].pct_change(periods=10) * 100
        result['price_momentum_20'] = df['y'].pct_change(periods=20) * 100

        # Price position in trading range
        result['high_20'] = df['High'].rolling(window=20).max() if 'High' in df.columns else df['y'].rolling(
            window=20).max()
        result['low_20'] = df['Low'].rolling(window=20).min() if 'Low' in df.columns else df['y'].rolling(
            window=20).min()

        price_range = result['high_20'] - result['low_20']
        result['price_position'] = (df['y'] - result['low_20']) / price_range
        result['price_position'] = result['price_position'].fillna(0.5)

        # Fill NaN values with reasonable defaults
        result = result.fillna(method='bfill').fillna(method='ffill')

    except Exception as e:
        print(f"⚠️ Warning: Could not calculate some technical indicators: {e}")

    return result


def analyze_market_sentiment(market_data: Dict, current_date: pd.Timestamp = None) -> Dict:
    """
    Analyze market sentiment using VIX and other indicators.
    """
    sentiment = {}

    try:
        # VIX Analysis (Fear Index)
        if 'vix' in market_data and not market_data['vix'].empty:
            recent_vix = market_data['vix'].tail(10).mean()
            current_vix = market_data['vix'].iloc[-1]

            if current_vix > 30:
                sentiment['fear_level'] = "High Fear"
                sentiment['fear_explanation'] = "Market shows high fear/volatility - may impact all stocks negatively"
                sentiment['fear_score'] = -2
            elif current_vix > 20:
                sentiment['fear_level'] = "Moderate Fear"
                sentiment['fear_explanation'] = "Market shows some concern - normal market conditions"
                sentiment['fear_score'] = -1
            else:
                sentiment['fear_level'] = "Low Fear"
                sentiment['fear_explanation'] = "Market confidence is high - good environment for stocks"
                sentiment['fear_score'] = 1

            sentiment['vix_value'] = current_vix

        # Market trend analysis (if SPY data available)
        if 'spy' in market_data and not market_data['spy'].empty:
            spy_recent = market_data['spy'].tail(20)
            spy_change = (spy_recent.iloc[-1] - spy_recent.iloc[0]) / spy_recent.iloc[0] * 100

            if spy_change > 2:
                sentiment['market_trend'] = "Bullish"
                sentiment['market_trend_score'] = 1
            elif spy_change < -2:
                sentiment['market_trend'] = "Bearish"
                sentiment['market_trend_score'] = -1
            else:
                sentiment['market_trend'] = "Neutral"
                sentiment['market_trend_score'] = 0

    except Exception as e:
        print(f"⚠️ Warning: Could not analyze market sentiment: {e}")

    return sentiment


# #############################
# ##     ENHANCED FORECASTING     ##
# #############################

def analyze_and_forecast_enhanced(df: pd.DataFrame, display_name: str, forecast_days: int,
                                  baseline_dividend: float, historical_dividends: pd.Series,
                                  show_plots: bool, currency: str, fifty_two_week_high: float,
                                  fifty_two_week_low: float, market_data: Dict) -> Tuple[
    Optional[pd.DataFrame], str, str, str, str, float, Dict]:
    """
    Enhanced version of original analyze_and_forecast function.
    Maintains all original functionality while adding enhanced analysis.
    """
    # Calculate technical indicators first
    enhanced_df = calculate_technical_indicators(df)

    # ORIGINAL PROPHET LOGIC (maintained exactly)
    # --- Data Transformation & Scaling ---
    df['y_log'] = np.log(df['y'])
    y_log_min = df['y_log'].min()
    y_log_max = df['y_log'].max()
    df['y_scaled'] = (df['y_log'] - y_log_min) / (y_log_max - y_log_min)

    # --- Model Training (original) ---
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_country_holidays(country_name='US')
    model.fit(df[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'}))

    # --- Forecasting (original) ---
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # --- Inverse Transform & Analysis (original) ---
    for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
        if col in forecast.columns:
            forecast[col] = forecast[col] * (y_log_max - y_log_min) + y_log_min

    current_price = df['y'].iloc[-1]
    future_prediction = forecast.iloc[-1]
    predicted_price = np.exp(future_prediction['yhat'])
    predicted_lower = np.exp(future_prediction['yhat_lower'])
    predicted_upper = np.exp(future_prediction['yhat_upper'])

    # ENHANCED ANALYSIS: Technical signals
    enhanced_analysis = analyze_technical_signals(enhanced_df, market_data, predicted_price, current_price)

    # ENHANCED ANALYSIS: Prediction confidence testing
    confidence_metrics = test_prediction_confidence(enhanced_df, forecast_days)

    # ORIGINAL ANALYSIS REPORT (maintained)
    signal, reason, risk_score, yield_status, projected_yield = print_analysis_report_enhanced(
        df, enhanced_df, forecast, display_name, current_price, predicted_price,
        predicted_lower, predicted_upper, baseline_dividend, historical_dividends,
        currency, fifty_two_week_high, fifty_two_week_low, enhanced_analysis, confidence_metrics
    )

    # ORIGINAL VISUALIZATION (maintained but enhanced)
    if show_plots:
        print("\n📈 Generating enhanced forecast plots...")
        try:
            create_enhanced_plots(df, enhanced_df, forecast, display_name, currency, market_data)
        except Exception as e:
            print(f"Could not generate plots. Error: {e}")

    # Return original format plus enhanced data
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

    return forecast, signal, reason, risk_score, yield_status, projected_yield, enhanced_analysis


def analyze_technical_signals(enhanced_df: pd.DataFrame, market_data: Dict,
                              predicted_price: float, current_price: float) -> Dict:
    """
    Analyze technical indicators to generate enhanced signals.
    """
    signals = {'positive': 0, 'negative': 0, 'details': []}

    try:
        latest = enhanced_df.iloc[-1]

        # Moving average signals
        if not pd.isna(latest.get('sma_20')) and not pd.isna(latest.get('sma_50')):
            if current_price > latest['sma_20'] > latest['sma_50']:
                signals['positive'] += 2
                signals['details'].append("Strong uptrend: Price above both 20-day and 50-day averages")
            elif current_price > latest['sma_20']:
                signals['positive'] += 1
                signals['details'].append("Mild uptrend: Price above 20-day average")
            elif current_price < latest['sma_20'] < latest['sma_50']:
                signals['negative'] += 2
                signals['details'].append("Strong downtrend: Price below both 20-day and 50-day averages")
            elif current_price < latest['sma_20']:
                signals['negative'] += 1
                signals['details'].append("Mild downtrend: Price below 20-day average")

        # Momentum signals
        if not pd.isna(latest.get('price_momentum_10')):
            momentum = latest['price_momentum_10']
            if momentum > 5:
                signals['positive'] += 1
                signals['details'].append(f"Strong positive momentum: +{momentum:.1f}% over 10 days")
            elif momentum < -5:
                signals['negative'] += 1
                signals['details'].append(f"Strong negative momentum: {momentum:.1f}% over 10 days")

        # Volume analysis
        if not pd.isna(latest.get('volume_ratio')):
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 1.5:
                signals['positive'] += 1
                signals['details'].append(f"High trading activity: {vol_ratio:.1f}x normal volume")
            elif vol_ratio < 0.5:
                signals['negative'] += 1
                signals['details'].append(f"Low trading activity: {vol_ratio:.1f}x normal volume")

        # Price position in range
        if not pd.isna(latest.get('price_position')):
            position = latest['price_position']
            if position > 0.8:
                signals['positive'] += 1
                signals['details'].append("Price near 20-day high - strong momentum")
            elif position < 0.2:
                signals['negative'] += 1
                signals['details'].append("Price near 20-day low - weak momentum")

        # Market sentiment integration
        market_sentiment = analyze_market_sentiment(market_data)
        if market_sentiment.get('fear_score'):
            fear_score = market_sentiment['fear_score']
            if fear_score > 0:
                signals['positive'] += fear_score
                signals['details'].append(f"Market sentiment: {market_sentiment['fear_level']}")
            else:
                signals['negative'] += abs(fear_score)
                signals['details'].append(f"Market sentiment: {market_sentiment['fear_level']}")

        # Prediction magnitude
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        if abs(price_change_pct) > 10:
            if price_change_pct > 0:
                signals['positive'] += 2
                signals['details'].append(f"Strong predicted growth: +{price_change_pct:.1f}%")
            else:
                signals['negative'] += 2
                signals['details'].append(f"Strong predicted decline: {price_change_pct:.1f}%")
        elif abs(price_change_pct) > 5:
            if price_change_pct > 0:
                signals['positive'] += 1
                signals['details'].append(f"Moderate predicted growth: +{price_change_pct:.1f}%")
            else:
                signals['negative'] += 1
                signals['details'].append(f"Moderate predicted decline: {price_change_pct:.1f}%")

    except Exception as e:
        print(f"⚠️ Warning: Could not complete technical analysis: {e}")

    return signals


def test_prediction_confidence(df: pd.DataFrame, forecast_days: int) -> Dict:
    """
    Test prediction accuracy on historical data to build confidence metrics.
    """
    confidence = {'tested': False, 'accuracy': 'Unknown', 'details': {}}

    try:
        if len(df) < 150:  # Need sufficient data for testing
            return confidence

        # Test on 2 different historical periods
        test_periods = [30, 60]
        all_errors = []

        for days in test_periods:
            for offset in [60, 120]:  # Test at different time points
                if len(df) <= days + offset:
                    continue

                # Split data
                train_end = len(df) - offset - days
                if train_end < 60:
                    continue

                train_data = df.iloc[:train_end].copy()
                actual_data = df.iloc[train_end:train_end + days].copy()

                if len(actual_data) < days:
                    continue

                try:
                    # Create mini-forecast
                    test_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                    test_model.fit(train_data[['ds', 'y']])

                    test_future = test_model.make_future_dataframe(periods=days)
                    test_forecast = test_model.predict(test_future)

                    # Compare predictions vs actual
                    predicted = test_forecast.tail(days)['yhat'].values
                    actual = actual_data['y'].values

                    if len(predicted) == len(actual):
                        errors = np.abs(predicted - actual) / actual * 100
                        all_errors.extend(errors)

                except:
                    continue

        if all_errors:
            avg_error = np.mean(all_errors)
            confidence['tested'] = True
            confidence['average_error_percent'] = avg_error
            confidence['tests_performed'] = len(all_errors)

            if avg_error < 5:
                confidence['accuracy'] = 'High'
            elif avg_error < 15:
                confidence['accuracy'] = 'Medium'
            else:
                confidence['accuracy'] = 'Low'

            confidence['details'] = {
                'avg_error_pct': avg_error,
                'error_range': [np.min(all_errors), np.max(all_errors)]
            }

    except Exception as e:
        print(f"⚠️ Warning: Could not test prediction confidence: {e}")

    return confidence


# #############################
# ##     ENHANCED REPORTING     ##
# #############################

def print_analysis_report_enhanced(df, enhanced_df, forecast, display_name, current_price, predicted_price,
                                   predicted_lower, predicted_upper, baseline_dividend, historical_dividends,
                                   currency, fifty_two_week_high, fifty_two_week_low, enhanced_analysis,
                                   confidence_metrics) -> Tuple[str, str, str, str, float]:
    """
    Enhanced version of original print_analysis_report.
    Maintains all original output while adding enhanced insights.
    """
    # ORIGINAL REPORT LOGIC (maintained exactly)
    last_known_date_obj = df['ds'].iloc[-1]
    predicted_date_obj = forecast['ds'].iloc[-1]

    # Original dividend analysis
    projected_yield = (baseline_dividend / predicted_price) * 100 if predicted_price > 0 else 0
    if projected_yield > YIELD_THRESHOLD_GOOD:
        yield_status = "Good"
    elif projected_yield > YIELD_THRESHOLD_AVG:
        yield_status = "Average"
    else:
        yield_status = "Low"

    # Original signal logic
    price_change_pct = ((predicted_price - current_price) / current_price) * 100
    if price_change_pct > SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📈 BUY", f"Price is forecast to rise by {price_change_pct:.2f}%."
    elif price_change_pct < -SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📉 SELL", f"Price is forecast to fall by {price_change_pct:.2f}%."
    else:
        signal, reason = "🤔 HOLD", f"Price is forecast to be stable (change: {price_change_pct:.2f}%)."

    # Original risk assessment
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

    # ENHANCED SIGNAL GENERATION
    enhanced_signal, enhanced_reason, enhanced_confidence = generate_enhanced_recommendation(
        signal, reason, enhanced_analysis, confidence_metrics, risk_metric
    )

    # ORIGINAL HEADER AND OUTPUT (maintained exactly)
    analysis_time = datetime.now(ZoneInfo("Asia/Tokyo")).strftime('%d-%b-%Y %I:%M:%S %p %Z')
    header = f"ENHANCED PROPHET-BASED STOCK ANALYSIS: {display_name}"
    print("\n" + "=" * 80)
    print(header.center(80))
    print(f"Analysis Time: {analysis_time}".center(80))
    print("=" * 80)
    print(f"Forecasting {FORECAST_DAYS} days (until {predicted_date_obj.strftime('%d-%b-%Y')}).\n")

    print(f"🔹 Last Market Data (as of {last_known_date_obj.strftime('%d-%b-%Y')}):")
    print(f"   - Closing Price: {currency} {current_price:,.2f}")

    # ORIGINAL Historical High/Low Information (maintained)
    print(f"   - Historical High / Low:")
    df_by_year = df.groupby(df['ds'].dt.year)

    for year, year_df in df_by_year:
        high_col = 'High' if 'High' in year_df.columns else 'y'
        low_col = 'Low' if 'Low' in year_df.columns else 'y'

        high_row = year_df.loc[year_df[high_col].idxmax()]
        year_high_price = high_row[high_col]
        year_high_date = high_row['ds'].strftime('%d-%b-%Y')

        low_row = year_df.loc[year_df[low_col].idxmin()]
        year_low_price = low_row[low_col]
        year_low_date = low_row['ds'].strftime('%d-%b-%Y')

        print(
            f"     - {year} High / Low: {currency} {year_high_price:,.2f} ({year_high_date}) / {currency} {year_low_price:,.2f} ({year_low_date})")

    print(f"\n   - 52-Week High / Low: {currency} {fifty_two_week_high:,.2f} / {currency} {fifty_two_week_low:,.2f}\n")

    # ORIGINAL Forecast section (maintained)
    print(f"🔮 Forecast for {predicted_date_obj.strftime('%d-%b-%Y')}:")
    print(f"   - Predicted Price: {currency} {predicted_price:,.2f}")
    print(f"   - Expected Range: {currency} {predicted_lower:,.2f} to {currency} {predicted_upper:,.2f}\n")

    # ENHANCED: Technical Analysis Summary
    print(f"📊 Technical Analysis Summary:")
    total_signals = enhanced_analysis['positive'] + enhanced_analysis['negative']
    net_signals = enhanced_analysis['positive'] - enhanced_analysis['negative']

    print(f"   - Positive Signals: {enhanced_analysis['positive']}")
    print(f"   - Negative Signals: {enhanced_analysis['negative']}")
    print(f"   - Net Signal Strength: {net_signals:+d}")

    if enhanced_analysis['details']:
        print(f"   - Key Technical Factors:")
        for detail in enhanced_analysis['details'][:3]:  # Show top 3 factors
            print(f"     • {detail}")
    print()

    # ENHANCED: Confidence Analysis
    if confidence_metrics['tested']:
        print(f"🎯 Prediction Confidence Analysis:")
        print(f"   - Historical Accuracy: {confidence_metrics['accuracy']}")
        print(f"   - Average Prediction Error: {confidence_metrics['average_error_percent']:.1f}%")
        print(f"   - Based on {confidence_metrics['tests_performed']} historical tests")
        if confidence_metrics['accuracy'] == 'High':
            print(f"   - Confidence Level: Our model has been quite accurate on similar predictions")
        elif confidence_metrics['accuracy'] == 'Medium':
            print(f"   - Confidence Level: Our model shows reasonable accuracy with some variation")
        else:
            print(f"   - Confidence Level: Predictions have higher uncertainty - use with caution")
        print()

    # ORIGINAL Dividend Analysis (maintained exactly)
    print(f"💰 Dividend Yield Analysis:")
    print(
        f"   - Projected Yield: {projected_yield:.2f}% ({currency} {baseline_dividend:.4f} per share) (based on a predicted price of {currency} {predicted_price:,.2f})")
    print(f"   - Projection Status: {yield_status}")

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
            print()
    else:
        print("   - Historical Payout: No dividends were paid in the tracked period.\n")

    # ENHANCED Buy/Sell Signal (combines original + enhanced)
    print(f"📊 Investment Recommendation:")
    print(f"   - Primary Signal: {enhanced_signal}")
    print(f"   - Enhanced Analysis: {enhanced_reason}")
    print(f"   - Original Prophet Signal: {signal} - {reason}")
    print(f"   - Confidence: {enhanced_confidence}\n")

    # ORIGINAL Risk Assessment (maintained)
    print(f"⚠️ Risk Assessment:")
    print(f"   - Model Uncertainty ({risk_metric:.2f}%) suggests the risk is: {risk_score}")

    # ENHANCED: Additional risk context
    if risk_metric > 30:
        print(f"   - High uncertainty suggests careful position sizing and close monitoring")
    elif risk_metric < 10:
        print(f"   - Low uncertainty suggests relatively stable prediction")

    print("=" * 80)

    # Return original format
    return enhanced_signal, enhanced_reason, risk_score, yield_status, projected_yield


def generate_enhanced_recommendation(original_signal: str, original_reason: str,
                                     enhanced_analysis: Dict, confidence_metrics: Dict, risk_metric: float) -> Tuple[
    str, str, str]:
    """
    Generate enhanced recommendation combining original Prophet analysis with technical analysis.
    """
    positive_signals = enhanced_analysis['positive']
    negative_signals = enhanced_analysis['negative']
    net_signals = positive_signals - negative_signals

    # Determine confidence level
    if confidence_metrics.get('tested', False):
        accuracy = confidence_metrics['accuracy']
        avg_error = confidence_metrics.get('average_error_percent', 20)

        if accuracy == 'High' and risk_metric < HIGH_CONFIDENCE_UNCERTAINTY:
            confidence = "High"
        elif accuracy in ['High', 'Medium'] and risk_metric < MEDIUM_CONFIDENCE_UNCERTAINTY:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        if risk_metric < HIGH_CONFIDENCE_UNCERTAINTY:
            confidence = "Medium"
        else:
            confidence = "Low"

    # Enhanced signal generation combining technical and Prophet analysis
    if net_signals >= TECHNICAL_SIGNAL_THRESHOLD and "BUY" in original_signal:
        enhanced_signal = "🚀 STRONG BUY"
        enhanced_reason = f"Both technical analysis ({positive_signals} positive vs {negative_signals} negative signals) and Prophet forecast strongly support buying"
    elif net_signals >= 1 and "BUY" in original_signal:
        enhanced_signal = "📈 BUY"
        enhanced_reason = f"Technical indicators ({positive_signals} positive signals) support the Prophet buy signal"
    elif net_signals <= -TECHNICAL_SIGNAL_THRESHOLD and "SELL" in original_signal:
        enhanced_signal = "🔻 STRONG SELL"
        enhanced_reason = f"Both technical analysis ({negative_signals} negative vs {positive_signals} positive signals) and Prophet forecast strongly support selling"
    elif net_signals <= -1 and "SELL" in original_signal:
        enhanced_signal = "📉 SELL"
        enhanced_reason = f"Technical indicators ({negative_signals} negative signals) support the Prophet sell signal"
    elif net_signals >= 2:
        enhanced_signal = "📈 TECHNICAL BUY"
        enhanced_reason = f"Strong technical signals ({positive_signals} positive vs {negative_signals} negative) suggest buying despite Prophet neutrality"
    elif net_signals <= -2:
        enhanced_signal = "📉 TECHNICAL SELL"
        enhanced_reason = f"Strong negative technical signals ({negative_signals} negative vs {positive_signals} positive) suggest selling despite Prophet neutrality"
    elif "BUY" in original_signal and net_signals >= 0:
        enhanced_signal = "📈 CAUTIOUS BUY"
        enhanced_reason = f"Prophet suggests buying but technical signals are mixed ({positive_signals} positive, {negative_signals} negative)"
    elif "SELL" in original_signal and net_signals <= 0:
        enhanced_signal = "📉 CAUTIOUS SELL"
        enhanced_reason = f"Prophet suggests selling and technical signals don't contradict ({positive_signals} positive, {negative_signals} negative)"
    else:
        enhanced_signal = "🤔 HOLD"
        enhanced_reason = f"Mixed signals from Prophet and technical analysis - wait for clearer direction"

    return enhanced_signal, enhanced_reason, confidence


# #############################
# ##     ENHANCED VISUALIZATION     ##
# #############################

def create_enhanced_plots(df: pd.DataFrame, enhanced_df: pd.DataFrame, forecast: pd.DataFrame,
                          display_name: str, currency: str, market_data: Dict):
    """
    Create enhanced plots combining original Prophet visualization with technical analysis.
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Original Prophet forecast with technical overlays
        ax1.plot(df['ds'], df['y'], 'k.', label='Historical Price', alpha=0.7)
        ax1.plot(forecast['ds'], np.exp(forecast['yhat']), ls='-', c='#0072B2', linewidth=2, label='Prophet Forecast')
        ax1.fill_between(forecast['ds'], np.exp(forecast['yhat_lower']), np.exp(forecast['yhat_upper']),
                         color='#0072B2', alpha=0.2, label='80% Confidence Interval')

        # Add technical indicators if available
        if 'sma_20' in enhanced_df.columns:
            ax1.plot(enhanced_df['ds'], enhanced_df['sma_20'], '--', color='orange', alpha=0.8, label='20-day SMA')
        if 'sma_50' in enhanced_df.columns:
            ax1.plot(enhanced_df['ds'], enhanced_df['sma_50'], '--', color='red', alpha=0.8, label='50-day SMA')

        ax1.set_title(f'Enhanced Price Forecast: {display_name}', size=14, fontweight='bold')
        ax1.set_ylabel(f"Price ({currency})", size=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volume analysis
        if 'Volume' in enhanced_df.columns:
            ax2.bar(enhanced_df['ds'], enhanced_df['Volume'], alpha=0.6, color='lightblue', width=1)
            if 'volume_sma_20' in enhanced_df.columns:
                ax2.plot(enhanced_df['ds'], enhanced_df['volume_sma_20'], 'red', linewidth=2, label='20-day Avg Volume')
            ax2.set_title('Trading Volume Analysis', size=14, fontweight='bold')
            ax2.set_ylabel('Volume', size=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Volume Analysis (N/A)', size=14)

        # Plot 3: Technical momentum indicators
        if 'price_momentum_10' in enhanced_df.columns:
            ax3.plot(enhanced_df['ds'], enhanced_df['price_momentum_10'], color='green', alpha=0.7,
                     label='10-day Momentum')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax3.axhline(y=-5, color='blue', linestyle='--', alpha=0.5, label='Oversold')
            ax3.fill_between(enhanced_df['ds'], 0, enhanced_df['price_momentum_10'],
                             where=(enhanced_df['price_momentum_10'] > 0), color='green', alpha=0.3)
            ax3.fill_between(enhanced_df['ds'], 0, enhanced_df['price_momentum_10'],
                             where=(enhanced_df['price_momentum_10'] < 0), color='red', alpha=0.3)
            ax3.set_title('Price Momentum (10-day % change)', size=14, fontweight='bold')
            ax3.set_ylabel('Momentum %', size=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Momentum data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Momentum Analysis (N/A)', size=14)

        # Plot 4: Market sentiment (VIX) if available
        if 'vix' in market_data and not market_data['vix'].empty:
            vix_data = market_data['vix']
            ax4.plot(vix_data.index, vix_data.values, color='purple', linewidth=2, label='VIX (Fear Index)')
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Low Fear')
            ax4.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='High Fear')
            ax4.fill_between(vix_data.index, 0, vix_data.values,
                             where=(vix_data.values > 30), color='red', alpha=0.3, label='Fear Zone')
            ax4.fill_between(vix_data.index, 0, vix_data.values,
                             where=(vix_data.values < 20), color='green', alpha=0.3, label='Confidence Zone')
            ax4.set_title('Market Sentiment (VIX Fear Index)', size=14, fontweight='bold')
            ax4.set_ylabel('VIX Level', size=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Market sentiment data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Market Sentiment (N/A)', size=14)

        plt.tight_layout()
        plt.show()

        # Create original Prophet components plot
        # Recreate model for components (since we need the fitted model)
        temp_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        temp_model.add_country_holidays(country_name='US')
        df_temp = df[['ds', 'y']].copy()
        df_temp['y'] = np.log(df_temp['y'])  # Use log transform like in main analysis
        temp_model.fit(df_temp)
        temp_future = temp_model.make_future_dataframe(periods=FORECAST_DAYS)
        temp_forecast = temp_model.predict(temp_future)

        temp_model.plot_components(temp_forecast)
        plt.suptitle('Prophet Model Components Analysis', size=16, fontweight='bold')
        plt.show()

        print("\n📊 CHART EXPLANATION:")
        print("   📈 Top-left: Price history, Prophet forecast, and moving averages")
        print("   📊 Top-right: Trading volume and average volume trends")
        print("   🔄 Bottom-left: Price momentum indicators (positive = upward pressure)")
        print("   😰 Bottom-right: Market fear index (VIX) - higher values mean more market anxiety")
        print("   📉 Components: Prophet's breakdown of trend, seasonality, and holiday effects")

    except Exception as e:
        # Fall back to original simple plot
        print(f"⚠️ Could not create enhanced plots, showing basic forecast: {e}")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['ds'], df['y'], 'k.', label='Historical Price')
        ax.plot(forecast['ds'], np.exp(forecast['yhat']), ls='-', c='#0072B2', label='Forecast')
        ax.fill_between(forecast['ds'], np.exp(forecast['yhat_lower']), np.exp(forecast['yhat_upper']),
                        color='#0072B2', alpha=0.2, label='80% Confidence Interval')
        ax.set_title(f'Price Forecast for {display_name}', size=16)
        ax.set_xlabel("Date", size=12)
        ax.set_ylabel(f"Price ({currency})", size=12)
        ax.legend()
        ax.grid(True, which='both', ls='--', lw=0.5)
        plt.show()


# #############################
# ##     ORIGINAL FUNCTIONS (MAINTAINED)     ##
# #############################

def display_portfolio_analysis(forecast: pd.DataFrame, buy_price: float, units: int, signal: str, reason: str,
                               currency: str):
    """
    Original portfolio analysis function - maintained exactly.
    """
    predicted_price = forecast.iloc[-1]['yhat']
    predicted_date = forecast.iloc[-1]['ds'].date().strftime('%d-%b-%Y')

    initial_cost = buy_price * units
    predicted_value = predicted_price * units
    potential_pl = predicted_value - initial_cost
    potential_pl_percent = (potential_pl / initial_cost) * 100 if initial_cost > 0 else 0

    if "BUY" in signal:
        action_signal = "📈 Accumulate More"
        justification = f"The forecast is positive ({reason}) and your position is projected to grow by {potential_pl_percent:+.2f}%. This could be a good opportunity to increase your holdings."
    elif "SELL" in signal:
        action_signal = "📉 Secure Profits / Cut Losses"
        if potential_pl > 0:
            justification = f"The forecast is negative ({reason}). Consider selling to secure your current profit of {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
        else:
            justification = f"The forecast is negative ({reason}). Consider selling to prevent further losses on your position, currently at {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
    else:
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
    Original investment profile analysis function - maintained exactly.
    """
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

    if yield_status == "Good":
        dividend_outlook = f"Excellent. The projected yield of {projected_yield:.2f}% is strong."
        if "Low" in risk_score or "Very Low" in risk_score:
            dividend_outlook += " Combined with a low risk score, this stock appears to be a solid candidate for dividend-focused investors."
        else:
            dividend_outlook += f" However, the {risk_score} risk level suggests investors should weigh the high yield against potential price volatility."
    elif yield_status == "Average":
        dividend_outlook = f"Average. The projected yield of {projected_yield:.2f}% provides some income, but may not be compelling for dedicated dividend investors."
    else:
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


# #############################
# ##     ENHANCED MAIN FUNCTION     ##
# #############################

def main():
    """
    Enhanced main function maintaining original CLI interface with enhanced functionality.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Prophet-based Stock Analysis Tool with Technical Analysis.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("ticker", help="The stock ticker symbol to analyze (e.g., 'AAPL', 'GOOGL').")
    parser.add_argument("--buy", type=float, help="Optional: The price at which you bought the stock.")
    parser.add_argument("--units", type=int, help="Optional: The number of units you own.")
    parser.add_argument('--no-plots', action='store_false', dest='show_plots',
                        help="Prevent forecast plots from being displayed.")
    parser.add_argument('--simple', action='store_true', dest='show_simple',
                        help="Show simplified analysis explanations alongside technical details.")
    parser.set_defaults(show_plots=SHOW_PLOTS, show_simple=SHOW_SIMPLE)

    args = parser.parse_args()

    if (args.buy is not None and args.units is None) or \
            (args.buy is None and args.units is not None):
        parser.error("Arguments --buy and --units must be used together.")

    print("🚀 Enhanced Prophet-Based Stock Analysis Tool")
    print("Combining time-series forecasting with technical analysis for better predictions\n")

    # Enhanced data fetching
    stock_df, stock_name, historical_dividends, baseline_dividend, currency, fifty_two_week_high, fifty_two_week_low, market_data = get_stock_data_enhanced(
        args.ticker, START_DATE_STR)

    if stock_df is not None and not stock_df.empty:
        display_name = f"{stock_name} ({args.ticker.upper()})"

        # Enhanced analysis
        forecast_data, signal, reason, risk_score, yield_status, projected_yield, enhanced_analysis = analyze_and_forecast_enhanced(
            stock_df, display_name, FORECAST_DAYS, baseline_dividend, historical_dividends,
            show_plots=args.show_plots, currency=currency, fifty_two_week_high=fifty_two_week_high,
            fifty_two_week_low=fifty_two_week_low, market_data=market_data
        )

        # Original portfolio or investment analysis
        if args.buy is not None and args.units is not None and forecast_data is not None:
            display_portfolio_analysis(forecast_data, args.buy, args.units, signal, reason, currency)

            # Enhanced: Additional portfolio insights
            if args.show_simple:
                print_simple_portfolio_explanation(forecast_data, args.buy, args.units, enhanced_analysis, currency)

        elif forecast_data is not None:
            print_investment_profile_analysis(
                stock_df, forecast_data, signal, reason, risk_score, yield_status, projected_yield
            )

            # Enhanced: Simple explanations if requested
            if args.show_simple:
                print_simple_investment_explanation(enhanced_analysis, signal, reason)


def print_simple_portfolio_explanation(forecast: pd.DataFrame, buy_price: float, units: int,
                                       enhanced_analysis: Dict, currency: str):
    """
    Provide simple explanations for portfolio analysis.
    """
    predicted_price = forecast.iloc[-1]['yhat']
    gain_loss = (predicted_price - buy_price) / buy_price * 100

    print(f"\n💡 SIMPLE EXPLANATION FOR YOUR INVESTMENT:")

    if gain_loss > 10:
        print(f"   🎉 Great news! Your investment might grow by about {gain_loss:.1f}%")
    elif gain_loss > 0:
        print(f"   📈 Good news! Small gains expected around {gain_loss:.1f}%")
    elif gain_loss > -10:
        print(f"   ⚠️  Small loss possible around {abs(gain_loss):.1f}% - monitor closely")
    else:
        print(f"   🔴 Significant loss possible around {abs(gain_loss):.1f}% - consider your options")

    print(
        f"   📊 This is based on {enhanced_analysis['positive']} positive and {enhanced_analysis['negative']} negative market signals")
    print(f"   📝 Remember: These are predictions based on patterns, not guarantees")


def print_simple_investment_explanation(enhanced_analysis: Dict, signal: str, reason: str):
    """
    Provide simple explanations for general investment analysis.
    """
    print(f"\n💡 SIMPLE EXPLANATION:")

    net_signals = enhanced_analysis['positive'] - enhanced_analysis['negative']

    if net_signals > 2:
        print(f"   🟢 Multiple positive signs suggest this might be a good buying opportunity")
    elif net_signals > 0:
        print(f"   🟡 Some positive signs, but be cautious - mixed signals present")
    elif net_signals < -2:
        print(f"   🔴 Multiple warning signs suggest avoiding or selling this stock")
    else:
        print(f"   🤔 Very mixed signals - probably best to wait and see what happens")

    print(f"   📈 Our computer model says: {reason}")
    print(
        f"   🔍 Market analysis found: {enhanced_analysis['positive']} positive and {enhanced_analysis['negative']} negative signals")

    if enhanced_analysis['details']:
        print(f"   📋 Main factors:")
        for i, detail in enumerate(enhanced_analysis['details'][:2], 1):
            print(f"      {i}. {detail}")


if __name__ == "__main__":
    main(), 0.0, 0.0, {}


# #############################
# ##     ENHANCED TECHNICAL ANALYSIS     ##
# #############################

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for enhanced analysis.
    Adds indicators without modifying original data structure.
    """
    result = df.copy()

    try:
        # Moving averages for trend analysis
        result['sma_20'] = df['y'].rolling(window=20).mean()
        result['sma_50'] = df['y'].rolling(window=50).mean()
        result['sma_200'] = df['y'].rolling(window=200).mean()

        # Daily returns and volatility
        result['daily_return'] = df['y'].pct_change()
        result['volatility_20'] = result['daily_return'].rolling(window=20).std() * np.sqrt(252)

        # Volume analysis (if available)
        if 'Volume' in df.columns:
            result['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = df['Volume'] / result['volume_sma_20']
            result['volume_ratio'] = result['volume_ratio'].fillna(1.0)

        # Momentum indicators
        result['price_momentum_10'] = df['y'].pct_change(periods=10) * 100
        result['price_momentum_20'] = df['y'].pct_change(periods=20) * 100

        # Price position in trading range
        result['high_20'] = df['High'].rolling(window=20).max() if 'High' in df.columns else df['y'].rolling(
            window=20).max()
        result['low_20'] = df['Low'].rolling(window=20).min() if 'Low' in df.columns else df['y'].rolling(
            window=20).min()

        price_range = result['high_20'] - result['low_20']
        result['price_position'] = (df['y'] - result['low_20']) / price_range
        result['price_position'] = result['price_position'].fillna(0.5)

        # Fill NaN values with reasonable defaults
        result = result.fillna(method='bfill').fillna(method='ffill')

    except Exception as e:
        print(f"⚠️ Warning: Could not calculate some technical indicators: {e}")

    return result


def analyze_market_sentiment(market_data: Dict, current_date: pd.Timestamp = None) -> Dict:
    """
    Analyze market sentiment using VIX and other indicators.
    """
    sentiment = {}

    try:
        # VIX Analysis (Fear Index)
        if 'vix' in market_data and not market_data['vix'].empty:
            recent_vix = market_data['vix'].tail(10).mean()
            current_vix = market_data['vix'].iloc[-1]

            if current_vix > 30:
                sentiment['fear_level'] = "High Fear"
                sentiment['fear_explanation'] = "Market shows high fear/volatility - may impact all stocks negatively"
                sentiment['fear_score'] = -2
            elif current_vix > 20:
                sentiment['fear_level'] = "Moderate Fear"
                sentiment['fear_explanation'] = "Market shows some concern - normal market conditions"
                sentiment['fear_score'] = -1
            else:
                sentiment['fear_level'] = "Low Fear"
                sentiment['fear_explanation'] = "Market confidence is high - good environment for stocks"
                sentiment['fear_score'] = 1

            sentiment['vix_value'] = current_vix

        # Market trend analysis (if SPY data available)
        if 'spy' in market_data and not market_data['spy'].empty:
            spy_recent = market_data['spy'].tail(20)
            spy_change = (spy_recent.iloc[-1] - spy_recent.iloc[0]) / spy_recent.iloc[0] * 100

            if spy_change > 2:
                sentiment['market_trend'] = "Bullish"
                sentiment['market_trend_score'] = 1
            elif spy_change < -2:
                sentiment['market_trend'] = "Bearish"
                sentiment['market_trend_score'] = -1
            else:
                sentiment['market_trend'] = "Neutral"
                sentiment['market_trend_score'] = 0

    except Exception as e:
        print(f"⚠️ Warning: Could not analyze market sentiment: {e}")

    return sentiment


# #############################
# ##     ENHANCED FORECASTING     ##
# #############################

def analyze_and_forecast_enhanced(df: pd.DataFrame, display_name: str, forecast_days: int,
                                  baseline_dividend: float, historical_dividends: pd.Series,
                                  show_plots: bool, currency: str, fifty_two_week_high: float,
                                  fifty_two_week_low: float, market_data: Dict) -> Tuple[
    Optional[pd.DataFrame], str, str, str, str, float, Dict]:
    """
    Enhanced version of original analyze_and_forecast function.
    Maintains all original functionality while adding enhanced analysis.
    """
    # Calculate technical indicators first
    enhanced_df = calculate_technical_indicators(df)

    # ORIGINAL PROPHET LOGIC (maintained exactly)
    # --- Data Transformation & Scaling ---
    df['y_log'] = np.log(df['y'])
    y_log_min = df['y_log'].min()
    y_log_max = df['y_log'].max()
    df['y_scaled'] = (df['y_log'] - y_log_min) / (y_log_max - y_log_min)

    # --- Model Training (original) ---
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_country_holidays(country_name='US')
    model.fit(df[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'}))

    # --- Forecasting (original) ---
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # --- Inverse Transform & Analysis (original) ---
    for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
        if col in forecast.columns:
            forecast[col] = forecast[col] * (y_log_max - y_log_min) + y_log_min

    current_price = df['y'].iloc[-1]
    future_prediction = forecast.iloc[-1]
    predicted_price = np.exp(future_prediction['yhat'])
    predicted_lower = np.exp(future_prediction['yhat_lower'])
    predicted_upper = np.exp(future_prediction['yhat_upper'])

    # ENHANCED ANALYSIS: Technical signals
    enhanced_analysis = analyze_technical_signals(enhanced_df, market_data, predicted_price, current_price)

    # ENHANCED ANALYSIS: Prediction confidence testing
    confidence_metrics = test_prediction_confidence(enhanced_df, forecast_days)

    # ORIGINAL ANALYSIS REPORT (maintained)
    signal, reason, risk_score, yield_status, projected_yield = print_analysis_report_enhanced(
        df, enhanced_df, forecast, display_name, current_price, predicted_price,
        predicted_lower, predicted_upper, baseline_dividend, historical_dividends,
        currency, fifty_two_week_high, fifty_two_week_low, enhanced_analysis, confidence_metrics
    )

    # ORIGINAL VISUALIZATION (maintained but enhanced)
    if show_plots:
        print("\n📈 Generating enhanced forecast plots...")
        try:
            create_enhanced_plots(df, enhanced_df, forecast, display_name, currency, market_data)
        except Exception as e:
            print(f"Could not generate plots. Error: {e}")

    # Return original format plus enhanced data
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

    return forecast, signal, reason, risk_score, yield_status, projected_yield, enhanced_analysis


def analyze_technical_signals(enhanced_df: pd.DataFrame, market_data: Dict,
                              predicted_price: float, current_price: float) -> Dict:
    """
    Analyze technical indicators to generate enhanced signals.
    """
    signals = {'positive': 0, 'negative': 0, 'details': []}

    try:
        latest = enhanced_df.iloc[-1]

        # Moving average signals
        if not pd.isna(latest.get('sma_20')) and not pd.isna(latest.get('sma_50')):
            if current_price > latest['sma_20'] > latest['sma_50']:
                signals['positive'] += 2
                signals['details'].append("Strong uptrend: Price above both 20-day and 50-day averages")
            elif current_price > latest['sma_20']:
                signals['positive'] += 1
                signals['details'].append("Mild uptrend: Price above 20-day average")
            elif current_price < latest['sma_20'] < latest['sma_50']:
                signals['negative'] += 2
                signals['details'].append("Strong downtrend: Price below both 20-day and 50-day averages")
            elif current_price < latest['sma_20']:
                signals['negative'] += 1
                signals['details'].append("Mild downtrend: Price below 20-day average")

        # Momentum signals
        if not pd.isna(latest.get('price_momentum_10')):
            momentum = latest['price_momentum_10']
            if momentum > 5:
                signals['positive'] += 1
                signals['details'].append(f"Strong positive momentum: +{momentum:.1f}% over 10 days")
            elif momentum < -5:
                signals['negative'] += 1
                signals['details'].append(f"Strong negative momentum: {momentum:.1f}% over 10 days")

        # Volume analysis
        if not pd.isna(latest.get('volume_ratio')):
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 1.5:
                signals['positive'] += 1
                signals['details'].append(f"High trading activity: {vol_ratio:.1f}x normal volume")
            elif vol_ratio < 0.5:
                signals['negative'] += 1
                signals['details'].append(f"Low trading activity: {vol_ratio:.1f}x normal volume")

        # Price position in range
        if not pd.isna(latest.get('price_position')):
            position = latest['price_position']
            if position > 0.8:
                signals['positive'] += 1
                signals['details'].append("Price near 20-day high - strong momentum")
            elif position < 0.2:
                signals['negative'] += 1
                signals['details'].append("Price near 20-day low - weak momentum")

        # Market sentiment integration
        market_sentiment = analyze_market_sentiment(market_data)
        if market_sentiment.get('fear_score'):
            fear_score = market_sentiment['fear_score']
            if fear_score > 0:
                signals['positive'] += fear_score
                signals['details'].append(f"Market sentiment: {market_sentiment['fear_level']}")
            else:
                signals['negative'] += abs(fear_score)
                signals['details'].append(f"Market sentiment: {market_sentiment['fear_level']}")

        # Prediction magnitude
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        if abs(price_change_pct) > 10:
            if price_change_pct > 0:
                signals['positive'] += 2
                signals['details'].append(f"Strong predicted growth: +{price_change_pct:.1f}%")
            else:
                signals['negative'] += 2
                signals['details'].append(f"Strong predicted decline: {price_change_pct:.1f}%")
        elif abs(price_change_pct) > 5:
            if price_change_pct > 0:
                signals['positive'] += 1
                signals['details'].append(f"Moderate predicted growth: +{price_change_pct:.1f}%")
            else:
                signals['negative'] += 1
                signals['details'].append(f"Moderate predicted decline: {price_change_pct:.1f}%")

    except Exception as e:
        print(f"⚠️ Warning: Could not complete technical analysis: {e}")

    return signals


def test_prediction_confidence(df: pd.DataFrame, forecast_days: int) -> Dict:
    """
    Test prediction accuracy on historical data to build confidence metrics.
    """
    confidence = {'tested': False, 'accuracy': 'Unknown', 'details': {}}

    try:
        if len(df) < 150:  # Need sufficient data for testing
            return confidence

        # Test on 2 different historical periods
        test_periods = [30, 60]
        all_errors = []

        for days in test_periods:
            for offset in [60, 120]:  # Test at different time points
                if len(df) <= days + offset:
                    continue

                # Split data
                train_end = len(df) - offset - days
                if train_end < 60:
                    continue

                train_data = df.iloc[:train_end].copy()
                actual_data = df.iloc[train_end:train_end + days].copy()

                if len(actual_data) < days:
                    continue

                try:
                    # Create mini-forecast
                    test_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                    test_model.fit(train_data[['ds', 'y']])

                    test_future = test_model.make_future_dataframe(periods=days)
                    test_forecast = test_model.predict(test_future)

                    # Compare predictions vs actual
                    predicted = test_forecast.tail(days)['yhat'].values
                    actual = actual_data['y'].values

                    if len(predicted) == len(actual):
                        errors = np.abs(predicted - actual) / actual * 100
                        all_errors.extend(errors)

                except:
                    continue

        if all_errors:
            avg_error = np.mean(all_errors)
            confidence['tested'] = True
            confidence['average_error_percent'] = avg_error
            confidence['tests_performed'] = len(all_errors)

            if avg_error < 5:
                confidence['accuracy'] = 'High'
            elif avg_error < 15:
                confidence['accuracy'] = 'Medium'
            else:
                confidence['accuracy'] = 'Low'

            confidence['details'] = {
                'avg_error_pct': avg_error,
                'error_range': [np.min(all_errors), np.max(all_errors)]
            }

    except Exception as e:
        print(f"⚠️ Warning: Could not test prediction confidence: {e}")

    return confidence


# #############################
# ##     ENHANCED REPORTING     ##
# #############################

def print_analysis_report_enhanced(df, enhanced_df, forecast, display_name, current_price, predicted_price,
                                   predicted_lower, predicted_upper, baseline_dividend, historical_dividends,
                                   currency, fifty_two_week_high, fifty_two_week_low, enhanced_analysis,
                                   confidence_metrics) -> Tuple[str, str, str, str, float]:
    """
    Enhanced version of original print_analysis_report.
    Maintains all original output while adding enhanced insights.
    """
    # ORIGINAL REPORT LOGIC (maintained exactly)
    last_known_date_obj = df['ds'].iloc[-1]
    predicted_date_obj = forecast['ds'].iloc[-1]

    # Original dividend analysis
    projected_yield = (baseline_dividend / predicted_price) * 100 if predicted_price > 0 else 0
    if projected_yield > YIELD_THRESHOLD_GOOD:
        yield_status = "Good"
    elif projected_yield > YIELD_THRESHOLD_AVG:
        yield_status = "Average"
    else:
        yield_status = "Low"

    # Original signal logic
    price_change_pct = ((predicted_price - current_price) / current_price) * 100
    if price_change_pct > SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📈 BUY", f"Price is forecast to rise by {price_change_pct:.2f}%."
    elif price_change_pct < -SIGNAL_PRICE_CHANGE_PCT:
        signal, reason = "📉 SELL", f"Price is forecast to fall by {price_change_pct:.2f}%."
    else:
        signal, reason = "🤔 HOLD", f"Price is forecast to be stable (change: {price_change_pct:.2f}%)."

    # Original risk assessment
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

    # ENHANCED SIGNAL GENERATION
    enhanced_signal, enhanced_reason, enhanced_confidence = generate_enhanced_recommendation(
        signal, reason, enhanced_analysis, confidence_metrics, risk_metric
    )

    # ORIGINAL HEADER AND OUTPUT (maintained exactly)
    analysis_time = datetime.now(ZoneInfo("Asia/Tokyo")).strftime('%d-%b-%Y %I:%M:%S %p %Z')
    header = f"ENHANCED PROPHET-BASED STOCK ANALYSIS: {display_name}"
    print("\n" + "=" * 80)
    print(header.center(80))
    print(f"Analysis Time: {analysis_time}".center(80))
    print("=" * 80)
    print(f"Forecasting {FORECAST_DAYS} days (until {predicted_date_obj.strftime('%d-%b-%Y')}).\n")

    print(f"🔹 Last Market Data (as of {last_known_date_obj.strftime('%d-%b-%Y')}):")
    print(f"   - Closing Price: {currency} {current_price:,.2f}")

    # ORIGINAL Historical High/Low Information (maintained)
    print(f"   - Historical High / Low:")
    df_by_year = df.groupby(df['ds'].dt.year)

    for year, year_df in df_by_year:
        high_col = 'High' if 'High' in year_df.columns else 'y'
        low_col = 'Low' if 'Low' in year_df.columns else 'y'

        high_row = year_df.loc[year_df[high_col].idxmax()]
        year_high_price = high_row[high_col]
        year_high_date = high_row['ds'].strftime('%d-%b-%Y')

        low_row = year_df.loc[year_df[low_col].idxmin()]
        year_low_price = low_row[low_col]
        year_low_date = low_row['ds'].strftime('%d-%b-%Y')

        print(
            f"     - {year} High / Low: {currency} {year_high_price:,.2f} ({year_high_date}) / {currency} {year_low_price:,.2f} ({year_low_date})")

    print(f"\n   - 52-Week High / Low: {currency} {fifty_two_week_high:,.2f} / {currency} {fifty_two_week_low:,.2f}\n")

    # ORIGINAL Forecast section (maintained)
    print(f"🔮 Forecast for {predicted_date_obj.strftime('%d-%b-%Y')}:")
    print(f"   - Predicted Price: {currency} {predicted_price:,.2f}")
    print(f"   - Expected Range: {currency} {predicted_lower:,.2f} to {currency} {predicted_upper:,.2f}\n")

    # ENHANCED: Technical Analysis Summary
    print(f"📊 Technical Analysis Summary:")
    total_signals = enhanced_analysis['positive'] + enhanced_analysis['negative']
    net_signals = enhanced_analysis['positive'] - enhanced_analysis['negative']

    print(f"   - Positive Signals: {enhanced_analysis['positive']}")
    print(f"   - Negative Signals: {enhanced_analysis['negative']}")
    print(f"   - Net Signal Strength: {net_signals:+d}")

    if enhanced_analysis['details']:
        print(f"   - Key Technical Factors:")
        for detail in enhanced_analysis['details'][:3]:  # Show top 3 factors
            print(f"     • {detail}")
    print()

    # ENHANCED: Confidence Analysis
    if confidence_metrics['tested']:
        print(f"🎯 Prediction Confidence Analysis:")
        print(f"   - Historical Accuracy: {confidence_metrics['accuracy']}")
        print(f"   - Average Prediction Error: {confidence_metrics['average_error_percent']:.1f}%")
        print(f"   - Based on {confidence_metrics['tests_performed']} historical tests")
        if confidence_metrics['accuracy'] == 'High':
            print(f"   - Confidence Level: Our model has been quite accurate on similar predictions")
        elif confidence_metrics['accuracy'] == 'Medium':
            print(f"   - Confidence Level: Our model shows reasonable accuracy with some variation")
        else:
            print(f"   - Confidence Level: Predictions have higher uncertainty - use with caution")
        print()

    # ORIGINAL Dividend Analysis (maintained exactly)
    print(f"💰 Dividend Yield Analysis:")
    print(
        f"   - Projected Yield: {projected_yield:.2f}% ({currency} {baseline_dividend:.4f} per share) (based on a predicted price of {currency} {predicted_price:,.2f})")
    print(f"   - Projection Status: {yield_status}")

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
            print()
    else:
        print("   - Historical Payout: No dividends were paid in the tracked period.\n")

    # ENHANCED Buy/Sell Signal (combines original + enhanced)
    print(f"📊 Investment Recommendation:")
    print(f"   - Primary Signal: {enhanced_signal}")
    print(f"   - Enhanced Analysis: {enhanced_reason}")
    print(f"   - Original Prophet Signal: {signal} - {reason}")
    print(f"   - Confidence: {enhanced_confidence}\n")

    # ORIGINAL Risk Assessment (maintained)
    print(f"⚠️ Risk Assessment:")
    print(f"   - Model Uncertainty ({risk_metric:.2f}%) suggests the risk is: {risk_score}")

    # ENHANCED: Additional risk context
    if risk_metric > 30:
        print(f"   - High uncertainty suggests careful position sizing and close monitoring")
    elif risk_metric < 10:
        print(f"   - Low uncertainty suggests relatively stable prediction")

    print("=" * 80)

    # Return original format
    return enhanced_signal, enhanced_reason, risk_score, yield_status, projected_yield


def generate_enhanced_recommendation(original_signal: str, original_reason: str,
                                     enhanced_analysis: Dict, confidence_metrics: Dict, risk_metric: float) -> Tuple[
    str, str, str]:
    """
    Generate enhanced recommendation combining original Prophet analysis with technical analysis.
    """
    positive_signals = enhanced_analysis['positive']
    negative_signals = enhanced_analysis['negative']
    net_signals = positive_signals - negative_signals

    # Determine confidence level
    if confidence_metrics.get('tested', False):
        accuracy = confidence_metrics['accuracy']
        avg_error = confidence_metrics.get('average_error_percent', 20)

        if accuracy == 'High' and risk_metric < HIGH_CONFIDENCE_UNCERTAINTY:
            confidence = "High"
        elif accuracy in ['High', 'Medium'] and risk_metric < MEDIUM_CONFIDENCE_UNCERTAINTY:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        if risk_metric < HIGH_CONFIDENCE_UNCERTAINTY:
            confidence = "Medium"
        else:
            confidence = "Low"

    # Enhanced signal generation combining technical and Prophet analysis
    if net_signals >= TECHNICAL_SIGNAL_THRESHOLD and "BUY" in original_signal:
        enhanced_signal = "🚀 STRONG BUY"
        enhanced_reason = f"Both technical analysis ({positive_signals} positive vs {negative_signals} negative signals) and Prophet forecast strongly support buying"
    elif net_signals >= 1 and "BUY" in original_signal:
        enhanced_signal = "📈 BUY"
        enhanced_reason = f"Technical indicators ({positive_signals} positive signals) support the Prophet buy signal"
    elif net_signals <= -TECHNICAL_SIGNAL_THRESHOLD and "SELL" in original_signal:
        enhanced_signal = "🔻 STRONG SELL"
        enhanced_reason = f"Both technical analysis ({negative_signals} negative vs {positive_signals} positive signals) and Prophet forecast strongly support selling"
    elif net_signals <= -1 and "SELL" in original_signal:
        enhanced_signal = "📉 SELL"
        enhanced_reason = f"Technical indicators ({negative_signals} negative signals) support the Prophet sell signal"
    elif net_signals >= 2:
        enhanced_signal = "📈 TECHNICAL BUY"
        enhanced_reason = f"Strong technical signals ({positive_signals} positive vs {negative_signals} negative) suggest buying despite Prophet neutrality"
    elif net_signals <= -2:
        enhanced_signal = "📉 TECHNICAL SELL"
        enhanced_reason = f"Strong negative technical signals ({negative_signals} negative vs {positive_signals} positive) suggest selling despite Prophet neutrality"
    elif "BUY" in original_signal and net_signals >= 0:
        enhanced_signal = "📈 CAUTIOUS BUY"
        enhanced_reason = f"Prophet suggests buying but technical signals are mixed ({positive_signals} positive, {negative_signals} negative)"
    elif "SELL" in original_signal and net_signals <= 0:
        enhanced_signal = "📉 CAUTIOUS SELL"
        enhanced_reason = f"Prophet suggests selling and technical signals don't contradict ({positive_signals} positive, {negative_signals} negative)"
    else:
        enhanced_signal = "🤔 HOLD"
        enhanced_reason = f"Mixed signals from Prophet and technical analysis - wait for clearer direction"

    return enhanced_signal, enhanced_reason, confidence


# #############################
# ##     ENHANCED VISUALIZATION     ##
# #############################

def create_enhanced_plots(df: pd.DataFrame, enhanced_df: pd.DataFrame, forecast: pd.DataFrame,
                          display_name: str, currency: str, market_data: Dict):
    """
    Create enhanced plots combining original Prophet visualization with technical analysis.
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Original Prophet forecast with technical overlays
        ax1.plot(df['ds'], df['y'], 'k.', label='Historical Price', alpha=0.7)
        ax1.plot(forecast['ds'], np.exp(forecast['yhat']), ls='-', c='#0072B2', linewidth=2, label='Prophet Forecast')
        ax1.fill_between(forecast['ds'], np.exp(forecast['yhat_lower']), np.exp(forecast['yhat_upper']),
                         color='#0072B2', alpha=0.2, label='80% Confidence Interval')

        # Add technical indicators if available
        if 'sma_20' in enhanced_df.columns:
            ax1.plot(enhanced_df['ds'], enhanced_df['sma_20'], '--', color='orange', alpha=0.8, label='20-day SMA')
        if 'sma_50' in enhanced_df.columns:
            ax1.plot(enhanced_df['ds'], enhanced_df['sma_50'], '--', color='red', alpha=0.8, label='50-day SMA')

        ax1.set_title(f'Enhanced Price Forecast: {display_name}', size=14, fontweight='bold')
        ax1.set_ylabel(f"Price ({currency})", size=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volume analysis
        if 'Volume' in enhanced_df.columns:
            ax2.bar(enhanced_df['ds'], enhanced_df['Volume'], alpha=0.6, color='lightblue', width=1)
            if 'volume_sma_20' in enhanced_df.columns:
                ax2.plot(enhanced_df['ds'], enhanced_df['volume_sma_20'], 'red', linewidth=2, label='20-day Avg Volume')
            ax2.set_title('Trading Volume Analysis', size=14, fontweight='bold')
            ax2.set_ylabel('Volume', size=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Volume Analysis (N/A)', size=14)

        # Plot 3: Technical momentum indicators
        if 'price_momentum_10' in enhanced_df.columns:
            ax3.plot(enhanced_df['ds'], enhanced_df['price_momentum_10'], color='green', alpha=0.7,
                     label='10-day Momentum')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax3.axhline(y=-5, color='blue', linestyle='--', alpha=0.5, label='Oversold')
            ax3.fill_between(enhanced_df['ds'], 0, enhanced_df['price_momentum_10'],
                             where=(enhanced_df['price_momentum_10'] > 0), color='green', alpha=0.3)
            ax3.fill_between(enhanced_df['ds'], 0, enhanced_df['price_momentum_10'],
                             where=(enhanced_df['price_momentum_10'] < 0), color='red', alpha=0.3)
            ax3.set_title('Price Momentum (10-day % change)', size=14, fontweight='bold')
            ax3.set_ylabel('Momentum %', size=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Momentum data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Momentum Analysis (N/A)', size=14)

        # Plot 4: Market sentiment (VIX) if available
        if 'vix' in market_data and not market_data['vix'].empty:
            vix_data = market_data['vix']
            ax4.plot(vix_data.index, vix_data.values, color='purple', linewidth=2, label='VIX (Fear Index)')
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Low Fear')
            ax4.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='High Fear')
            ax4.fill_between(vix_data.index, 0, vix_data.values,
                             where=(vix_data.values > 30), color='red', alpha=0.3, label='Fear Zone')
            ax4.fill_between(vix_data.index, 0, vix_data.values,
                             where=(vix_data.values < 20), color='green', alpha=0.3, label='Confidence Zone')
            ax4.set_title('Market Sentiment (VIX Fear Index)', size=14, fontweight='bold')
            ax4.set_ylabel('VIX Level', size=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Market sentiment data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Market Sentiment (N/A)', size=14)

        plt.tight_layout()
        plt.show()

        # Create original Prophet components plot
        # Recreate model for components (since we need the fitted model)
        temp_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        temp_model.add_country_holidays(country_name='US')
        df_temp = df[['ds', 'y']].copy()
        df_temp['y'] = np.log(df_temp['y'])  # Use log transform like in main analysis
        temp_model.fit(df_temp)
        temp_future = temp_model.make_future_dataframe(periods=FORECAST_DAYS)
        temp_forecast = temp_model.predict(temp_future)

        temp_model.plot_components(temp_forecast)
        plt.suptitle('Prophet Model Components Analysis', size=16, fontweight='bold')
        plt.show()

        print("\n📊 CHART EXPLANATION:")
        print("   📈 Top-left: Price history, Prophet forecast, and moving averages")
        print("   📊 Top-right: Trading volume and average volume trends")
        print("   🔄 Bottom-left: Price momentum indicators (positive = upward pressure)")
        print("   😰 Bottom-right: Market fear index (VIX) - higher values mean more market anxiety")
        print("   📉 Components: Prophet's breakdown of trend, seasonality, and holiday effects")

    except Exception as e:
        # Fall back to original simple plot
        print(f"⚠️ Could not create enhanced plots, showing basic forecast: {e}")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['ds'], df['y'], 'k.', label='Historical Price')
        ax.plot(forecast['ds'], np.exp(forecast['yhat']), ls='-', c='#0072B2', label='Forecast')
        ax.fill_between(forecast['ds'], np.exp(forecast['yhat_lower']), np.exp(forecast['yhat_upper']),
                        color='#0072B2', alpha=0.2, label='80% Confidence Interval')
        ax.set_title(f'Price Forecast for {display_name}', size=16)
        ax.set_xlabel("Date", size=12)
        ax.set_ylabel(f"Price ({currency})", size=12)
        ax.legend()
        ax.grid(True, which='both', ls='--', lw=0.5)
        plt.show()


# #############################
# ##     ORIGINAL FUNCTIONS (MAINTAINED)     ##
# #############################

def display_portfolio_analysis(forecast: pd.DataFrame, buy_price: float, units: int, signal: str, reason: str,
                               currency: str):
    """
    Original portfolio analysis function - maintained exactly.
    """
    predicted_price = forecast.iloc[-1]['yhat']
    predicted_date = forecast.iloc[-1]['ds'].date().strftime('%d-%b-%Y')

    initial_cost = buy_price * units
    predicted_value = predicted_price * units
    potential_pl = predicted_value - initial_cost
    potential_pl_percent = (potential_pl / initial_cost) * 100 if initial_cost > 0 else 0

    if "BUY" in signal:
        action_signal = "📈 Accumulate More"
        justification = f"The forecast is positive ({reason}) and your position is projected to grow by {potential_pl_percent:+.2f}%. This could be a good opportunity to increase your holdings."
    elif "SELL" in signal:
        action_signal = "📉 Secure Profits / Cut Losses"
        if potential_pl > 0:
            justification = f"The forecast is negative ({reason}). Consider selling to secure your current profit of {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
        else:
            justification = f"The forecast is negative ({reason}). Consider selling to prevent further losses on your position, currently at {currency} {potential_pl:,.2f} ({potential_pl_percent:+.2f}%)."
    else:
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
    Original investment profile analysis function - maintained exactly.
    """
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

    if yield_status == "Good":
        dividend_outlook = f"Excellent. The projected yield of {projected_yield:.2f}% is strong."
        if "Low" in risk_score or "Very Low" in risk_score:
            dividend_outlook += " Combined with a low risk score, this stock appears to be a solid candidate for dividend-focused investors."
        else:
            dividend_outlook += f" However, the {risk_score} risk level suggests investors should weigh the high yield against potential price volatility."
    elif yield_status == "Average":
        dividend_outlook = f"Average. The projected yield of {projected_yield:.2f}% provides some income, but may not be compelling for dedicated dividend investors."
    else:
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


# #############################
# ##     ENHANCED MAIN FUNCTION     ##
# #############################

def main():
    """
    Enhanced main function maintaining original CLI interface with enhanced functionality.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Prophet-based Stock Analysis Tool with Technical Analysis.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("ticker", help="The stock ticker symbol to analyze (e.g., 'AAPL', 'GOOGL').")
    parser.add_argument("--buy", type=float, help="Optional: The price at which you bought the stock.")
    parser.add_argument("--units", type=int, help="Optional: The number of units you own.")
    parser.add_argument('--no-plots', action='store_false', dest='show_plots',
                        help="Prevent forecast plots from being displayed.")
    parser.add_argument('--simple', action='store_true', dest='show_simple',
                        help="Show simplified analysis explanations alongside technical details.")
    parser.set_defaults(show_plots=SHOW_PLOTS, show_simple=SHOW_SIMPLE)

    args = parser.parse_args()

    if (args.buy is not None and args.units is None) or \
            (args.buy is None and args.units is not None):
        parser.error("Arguments --buy and --units must be used together.")

    print("🚀 Enhanced Prophet-Based Stock Analysis Tool")
    print("Combining time-series forecasting with technical analysis for better predictions\n")

    # Enhanced data fetching
    stock_df, stock_name, historical_dividends, baseline_dividend, currency, fifty_two_week_high, fifty_two_week_low, market_data = get_stock_data_enhanced(
        args.ticker, START_DATE_STR)

    if stock_df is not None and not stock_df.empty:
        display_name = f"{stock_name} ({args.ticker.upper()})"

        # Enhanced analysis
        forecast_data, signal, reason, risk_score, yield_status, projected_yield, enhanced_analysis = analyze_and_forecast_enhanced(
            stock_df, display_name, FORECAST_DAYS, baseline_dividend, historical_dividends,
            show_plots=args.show_plots, currency=currency, fifty_two_week_high=fifty_two_week_high,
            fifty_two_week_low=fifty_two_week_low, market_data=market_data
        )

        # Original portfolio or investment analysis
        if args.buy is not None and args.units is not None and forecast_data is not None:
            display_portfolio_analysis(forecast_data, args.buy, args.units, signal, reason, currency)

            # Enhanced: Additional portfolio insights
            if args.show_simple:
                print_simple_portfolio_explanation(forecast_data, args.buy, args.units, enhanced_analysis, currency)

        elif forecast_data is not None:
            print_investment_profile_analysis(
                stock_df, forecast_data, signal, reason, risk_score, yield_status, projected_yield
            )

            # Enhanced: Simple explanations if requested
            if args.show_simple:
                print_simple_investment_explanation(enhanced_analysis, signal, reason)


def print_simple_portfolio_explanation(forecast: pd.DataFrame, buy_price: float, units: int,
                                       enhanced_analysis: Dict, currency: str):
    """
    Provide simple explanations for portfolio analysis.
    """
    predicted_price = forecast.iloc[-1]['yhat']
    gain_loss = (predicted_price - buy_price) / buy_price * 100

    print(f"\n💡 SIMPLE EXPLANATION FOR YOUR INVESTMENT:")

    if gain_loss > 10:
        print(f"   🎉 Great news! Your investment might grow by about {gain_loss:.1f}%")
    elif gain_loss > 0:
        print(f"   📈 Good news! Small gains expected around {gain_loss:.1f}%")
    elif gain_loss > -10:
        print(f"   ⚠️  Small loss possible around {abs(gain_loss):.1f}% - monitor closely")
    else:
        print(f"   🔴 Significant loss possible around {abs(gain_loss):.1f}% - consider your options")

    print(
        f"   📊 This is based on {enhanced_analysis['positive']} positive and {enhanced_analysis['negative']} negative market signals")
    print(f"   📝 Remember: These are predictions based on patterns, not guarantees")


def print_simple_investment_explanation(enhanced_analysis: Dict, signal: str, reason: str):
    """
    Provide simple explanations for general investment analysis.
    """
    print(f"\n💡 SIMPLE EXPLANATION:")

    net_signals = enhanced_analysis['positive'] - enhanced_analysis['negative']

    if net_signals > 2:
        print(f"   🟢 Multiple positive signs suggest this might be a good buying opportunity")
    elif net_signals > 0:
        print(f"   🟡 Some positive signs, but be cautious - mixed signals present")
    elif net_signals < -2:
        print(f"   🔴 Multiple warning signs suggest avoiding or selling this stock")
    else:
        print(f"   🤔 Very mixed signals - probably best to wait and see what happens")

    print(f"   📈 Our computer model says: {reason}")
    print(
        f"   🔍 Market analysis found: {enhanced_analysis['positive']} positive and {enhanced_analysis['negative']} negative signals")

    if enhanced_analysis['details']:
        print(f"   📋 Main factors:")
        for i, detail in enumerate(enhanced_analysis['details'][:2], 1):
            print(f"      {i}. {detail}")


if __name__ == "__main__":
    main()