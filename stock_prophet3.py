# -*- coding: utf-8 -*-
"""
Enhanced Stock Analysis Tool (Simplified Version)

This tool analyzes stocks and predicts future prices using multiple methods
to give you better, more reliable predictions. It explains everything in
simple terms that anyone can understand.
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

# Turn off technical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Settings
START_DATE_STR = '2020-01-01'  # Get 4+ years of data for better patterns
FORECAST_DAYS = 90  # Predict 3 months ahead
SHOW_PLOTS = True


class SimpleStockAnalyzer:
    """A stock analyzer that explains things in simple terms"""

    def get_stock_data(self, ticker: str) -> Dict:
        """Get stock data and related market information"""
        try:
            print(f"📊 Getting data for {ticker}...")

            # Main stock data
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            stock_data = stock.history(start=START_DATE_STR, auto_adjust=True)

            if stock_data.empty:
                return {"error": f"Could not find data for {ticker}"}

            # Clean the stock data
            stock_data = stock_data.dropna()
            stock_data = stock_data[stock_data['Close'] > 0]  # Remove invalid prices

            if len(stock_data) < 60:
                return {"error": f"Not enough data for {ticker}. Found {len(stock_data)} days, need at least 60 days."}

            # Get market data for comparison (simplified)
            market_data = {}

            # Fear index (VIX) - shows if investors are scared
            try:
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(start=START_DATE_STR)['Close']
                if len(vix_data) > 0:
                    market_data['fear_index'] = vix_data
                else:
                    market_data['fear_index'] = None
            except:
                market_data['fear_index'] = None

            company_name = stock_info.get('longName', stock_info.get('shortName', ticker))
            print(f"✅ Got {len(stock_data)} days of data for {company_name}")

            return {
                'stock_data': stock_data,
                'stock_info': stock_info,
                'market_data': market_data,
                'dividends': None  # Simplified for now
            }

        except Exception as e:
            return {"error": f"Error getting data for {ticker}: {e}"}

    def calculate_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple indicators that help predict price movements"""
        result = df.copy()

        try:
            # Moving averages (smooth out daily price jumps)
            result['avg_20_days'] = df['Close'].rolling(window=20).mean()
            result['avg_50_days'] = df['Close'].rolling(window=50).mean()

            # Daily returns for volatility
            result['daily_change'] = df['Close'].pct_change()

            # Volume trend (are more people buying/selling than usual?)
            result['avg_volume'] = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = df['Volume'] / result['avg_volume']
            result['volume_ratio'] = result['volume_ratio'].fillna(1.0)  # Fill NaN with 1.0

            # Price momentum (how fast price is changing)
            result['momentum'] = df['Close'].pct_change(periods=10) * 100
            result['momentum'] = result['momentum'].fillna(0)  # Fill NaN with 0

            # Price position in recent range
            result['high_20'] = df['High'].rolling(window=20).max()
            result['low_20'] = df['Low'].rolling(window=20).min()
            price_range = result['high_20'] - result['low_20']
            result['price_position'] = (df['Close'] - result['low_20']) / price_range
            result['price_position'] = result['price_position'].fillna(0.5)  # Fill NaN with 0.5

        except Exception as e:
            print(f"⚠️ Warning: Could not calculate some indicators: {e}")
            # Add default values for missing indicators
            for col in ['avg_20_days', 'avg_50_days', 'daily_change', 'volume_ratio', 'momentum', 'price_position']:
                if col not in result.columns:
                    if col in ['volume_ratio']:
                        result[col] = 1.0
                    elif col in ['momentum']:
                        result[col] = 0.0
                    elif col in ['price_position']:
                        result[col] = 0.5
                    else:
                        result[col] = result['Close']  # Default to close price for averages

        return result

    def analyze_current_situation(self, df: pd.DataFrame, market_data: Dict) -> Dict:
        """Analyze what's happening with the stock right now"""
        try:
            # First add the indicators to the dataframe
            enhanced_df = self.calculate_simple_indicators(df)
            latest = enhanced_df.iloc[-1]
            recent_20_days = enhanced_df.tail(20)

            situation = {}

            # Price trend
            current_price = latest['Close']
            avg_20 = latest.get('avg_20_days', current_price)
            avg_50 = latest.get('avg_50_days', current_price)

            if pd.isna(avg_20):
                avg_20 = current_price
            if pd.isna(avg_50):
                avg_50 = current_price

            if current_price > avg_20 > avg_50:
                situation['trend'] = "Strong upward trend"
                situation['trend_explanation'] = "Price is above both short and long-term averages"
            elif current_price > avg_20:
                situation['trend'] = "Mild upward trend"
                situation['trend_explanation'] = "Price is above recent average but trend is weakening"
            elif current_price < avg_20 < avg_50:
                situation['trend'] = "Strong downward trend"
                situation['trend_explanation'] = "Price is below both short and long-term averages"
            else:
                situation['trend'] = "Sideways/Mixed trend"
                situation['trend_explanation'] = "Price is moving sideways without clear direction"

            # Volatility (how much the price jumps around)
            if 'daily_change' in recent_20_days.columns:
                recent_volatility = recent_20_days['daily_change'].std() * np.sqrt(252) * 100
                if pd.isna(recent_volatility):
                    recent_volatility = 20  # Default moderate volatility
            else:
                recent_volatility = 20

            if recent_volatility > 40:
                situation['volatility'] = "Very jumpy"
                situation['volatility_explanation'] = "Price swings a lot daily - higher risk"
            elif recent_volatility > 25:
                situation['volatility'] = "Moderately jumpy"
                situation['volatility_explanation'] = "Some price swings - normal risk"
            else:
                situation['volatility'] = "Stable"
                situation['volatility_explanation'] = "Price moves smoothly - lower risk"

            # Volume (trading activity)
            if 'volume_ratio' in recent_20_days.columns:
                recent_volume_avg = recent_20_days['volume_ratio'].mean()
                if pd.isna(recent_volume_avg):
                    recent_volume_avg = 1.0
            else:
                recent_volume_avg = 1.0

            if recent_volume_avg > 1.5:
                situation['activity'] = "Very active trading"
                situation['activity_explanation'] = "Much more trading than usual - big interest"
            elif recent_volume_avg > 1.2:
                situation['activity'] = "Active trading"
                situation['activity_explanation'] = "More trading than usual - growing interest"
            else:
                situation['activity'] = "Normal trading"
                situation['activity_explanation'] = "Regular trading levels"

            # Market fear level
            if market_data.get('fear_index') is not None and len(market_data['fear_index']) > 0:
                try:
                    recent_fear = market_data['fear_index'].tail(5).mean()
                    if pd.isna(recent_fear):
                        recent_fear = 20  # Default neutral fear

                    if recent_fear > 30:
                        situation['market_mood'] = "Investors are worried"
                        situation['market_explanation'] = "High fear in overall market - affects all stocks"
                    elif recent_fear < 15:
                        situation['market_mood'] = "Investors are confident"
                        situation['market_explanation'] = "Low fear in market - good for stocks generally"
                    else:
                        situation['market_mood'] = "Normal investor sentiment"
                        situation['market_explanation'] = "Market fear levels are normal"
                except:
                    pass  # Skip market mood if VIX data is problematic

            return situation

        except Exception as e:
            print(f"⚠️ Warning: Could not complete situation analysis: {e}")
            return {
                'trend': 'Unable to determine',
                'trend_explanation': 'Insufficient data for trend analysis',
                'volatility': 'Unknown',
                'volatility_explanation': 'Could not calculate volatility',
                'activity': 'Unknown',
                'activity_explanation': 'Could not analyze trading activity'
            }

    def create_basic_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a single, reliable prediction"""
        try:
            # Prepare data for Prophet
            prophet_df = df[['Close']].reset_index()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

            # Clean data
            prophet_df = prophet_df.dropna()
            prophet_df = prophet_df[prophet_df['y'] > 0]

            if len(prophet_df) < 30:
                raise ValueError(f"Not enough clean data: {len(prophet_df)} days")

            print(f"   Using {len(prophet_df)} days of data for prediction")

            # Create a simple, reliable Prophet model
            model = Prophet(
                changepoint_prior_scale=0.1,  # Balanced sensitivity
                seasonality_prior_scale=3.0,  # Moderate seasonality
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.8
            )

            # Add US holidays (works for most markets)
            try:
                model.add_country_holidays(country_name='US')
            except:
                pass  # Skip holidays if there's an issue

            # Fit the model
            model.fit(prophet_df)

            # Make prediction
            future = model.make_future_dataframe(periods=FORECAST_DAYS)
            forecast = model.predict(future)

            return forecast

        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

    def test_prediction_accuracy(self, df: pd.DataFrame) -> Dict:
        """Test how accurate our predictions have been on past data"""
        try:
            print("   Testing prediction accuracy...")

            results = {}
            test_periods = [30, 60]  # Test 1 and 2 month predictions

            for days in test_periods:
                if len(df) <= days + 90:  # Need enough data
                    continue

                errors = []

                # Test 2 different periods
                for i in range(2):
                    test_start = len(df) - days - (i * 30)
                    if test_start <= 60:
                        break

                    # Use data up to test_start for training
                    train_data = df.iloc[:test_start].copy()
                    actual_data = df.iloc[test_start:test_start + days].copy()

                    if len(actual_data) < days:
                        continue

                    try:
                        # Make prediction using basic method
                        train_prophet = train_data[['Close']].reset_index()
                        train_prophet.columns = ['ds', 'y']
                        train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
                        train_prophet = train_prophet.dropna()

                        if len(train_prophet) < 30:
                            continue

                        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                        model.fit(train_prophet)

                        future = model.make_future_dataframe(periods=days)
                        forecast = model.predict(future)

                        # Compare prediction vs actual
                        predicted_prices = forecast.tail(days)['yhat'].values
                        actual_prices = actual_data['Close'].values

                        if len(predicted_prices) == len(actual_prices):
                            # Calculate average error
                            avg_error = np.mean(np.abs(predicted_prices - actual_prices))
                            errors.append(avg_error)

                    except:
                        continue

                if errors:
                    avg_error = np.mean(errors)
                    current_price = df['Close'].iloc[-1]
                    error_percentage = (avg_error / current_price) * 100

                    results[f'{days}_days'] = {
                        'avg_error_dollars': avg_error,
                        'error_percentage': error_percentage,
                        'tests_run': len(errors)
                    }

            return results

        except Exception as e:
            print(f"⚠️ Warning: Could not test accuracy: {e}")
            return {}

    def analyze_stock(self, ticker: str) -> Dict:
        """Main function to analyze a stock completely"""
        print(f"🔍 Starting analysis for {ticker.upper()}...")

        # Get all the data
        data_result = self.get_stock_data(ticker.upper())
        if "error" in data_result:
            return data_result

        stock_data = data_result['stock_data']
        stock_info = data_result['stock_info']
        market_data = data_result['market_data']

        # Analyze current situation
        print("📊 Analyzing current situation...")
        current_situation = self.analyze_current_situation(stock_data, market_data)

        # Create prediction
        print("🔮 Creating price prediction...")
        try:
            forecast = self.create_basic_prediction(stock_data)
        except Exception as e:
            return {"error": f"Could not create prediction: {e}"}

        # Test accuracy
        print("🎯 Testing prediction accuracy...")
        accuracy_results = self.test_prediction_accuracy(stock_data)

        return {
            'stock_info': stock_info,
            'stock_data': stock_data,
            'current_situation': current_situation,
            'forecast': forecast,
            'accuracy_test': accuracy_results,
            'market_data': market_data
        }


def create_simple_recommendation(results: Dict) -> Dict:
    """Create a simple buy/sell/hold recommendation"""

    current_situation = results['current_situation']
    forecast = results['forecast']
    stock_data = results['stock_data']

    # Get current and predicted prices
    current_price = stock_data['Close'].iloc[-1]
    predicted_price = forecast.iloc[-1]['yhat']
    price_change_percent = ((predicted_price - current_price) / current_price) * 100

    # Count positive and negative factors
    positive_signals = 0
    negative_signals = 0

    # Check trend
    if "upward" in current_situation.get('trend', '').lower():
        positive_signals += 2 if "strong" in current_situation['trend'].lower() else 1
    elif "downward" in current_situation.get('trend', '').lower():
        negative_signals += 2 if "strong" in current_situation['trend'].lower() else 1

    # Check predicted price change
    if price_change_percent > 5:
        positive_signals += 2
    elif price_change_percent > 2:
        positive_signals += 1
    elif price_change_percent < -5:
        negative_signals += 2
    elif price_change_percent < -2:
        negative_signals += 1

    # Check volatility (high volatility = higher risk)
    if "stable" in current_situation.get('volatility', '').lower():
        positive_signals += 1
    elif "very jumpy" in current_situation.get('volatility', '').lower():
        negative_signals += 1

    # Check market mood if available
    if current_situation.get('market_mood'):
        if "confident" in current_situation['market_mood'].lower():
            positive_signals += 1
        elif "worried" in current_situation['market_mood'].lower():
            negative_signals += 1

    # Calculate confidence range
    confidence_range = forecast.iloc[-1]['yhat_upper'] - forecast.iloc[-1]['yhat_lower']
    uncertainty_percent = (confidence_range / predicted_price) * 100

    # Make recommendation
    net_signals = positive_signals - negative_signals

    if net_signals >= 3:
        recommendation = "👍 CONSIDER BUYING"
        reasoning = "Multiple positive signals suggest good potential for growth"
        confidence = "High" if uncertainty_percent < 15 else "Medium"
    elif net_signals >= 1:
        recommendation = "👍 MILD BUY"
        reasoning = "Some positive signals, but with caution"
        confidence = "Medium" if uncertainty_percent < 20 else "Low"
    elif net_signals <= -3:
        recommendation = "👎 CONSIDER SELLING"
        reasoning = "Multiple negative signals suggest potential decline"
        confidence = "High" if uncertainty_percent < 15 else "Medium"
    elif net_signals <= -1:
        recommendation = "👎 MILD SELL"
        reasoning = "Some negative signals suggest caution"
        confidence = "Medium" if uncertainty_percent < 20 else "Low"
    else:
        recommendation = "🤔 HOLD/WAIT"
        reasoning = "Mixed signals suggest waiting for clearer direction"
        confidence = "Low" if uncertainty_percent > 25 else "Medium"

    return {
        'recommendation': recommendation,
        'reasoning': reasoning,
        'confidence': confidence,
        'positive_signals': positive_signals,
        'negative_signals': negative_signals,
        'predicted_change': price_change_percent,
        'uncertainty_percent': uncertainty_percent
    }


def display_simple_results(results: Dict, ticker: str):
    """Display results in very simple terms"""

    stock_info = results['stock_info']
    current_situation = results['current_situation']
    forecast = results['forecast']
    stock_data = results['stock_data']
    accuracy = results['accuracy_test']

    company_name = stock_info.get('longName', stock_info.get('shortName', ticker))
    current_price = stock_data['Close'].iloc[-1]
    predicted_price = forecast.iloc[-1]['yhat']
    predicted_low = forecast.iloc[-1]['yhat_lower']
    predicted_high = forecast.iloc[-1]['yhat_upper']

    print("\n" + "=" * 80)
    print(f"📈 STOCK ANALYSIS REPORT: {company_name} ({ticker.upper()})")
    print("=" * 80)

    # Current situation
    print("\n🎯 CURRENT SITUATION:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Trend: {current_situation.get('trend', 'Unknown')}")
    print(f"   → {current_situation.get('trend_explanation', 'No explanation available')}")
    print(f"   Price Stability: {current_situation.get('volatility', 'Unknown')}")
    print(f"   → {current_situation.get('volatility_explanation', 'No explanation available')}")
    print(f"   Trading Activity: {current_situation.get('activity', 'Unknown')}")
    print(f"   → {current_situation.get('activity_explanation', 'No explanation available')}")

    if current_situation.get('market_mood'):
        print(f"   Overall Market: {current_situation['market_mood']}")
        print(f"   → {current_situation['market_explanation']}")

    # Prediction
    predicted_date = forecast.iloc[-1]['ds'].strftime('%B %d, %Y')
    price_change = predicted_price - current_price
    price_change_percent = (price_change / current_price) * 100

    print(f"\n🔮 PRICE PREDICTION (for {predicted_date}):")
    print(f"   Predicted Price: ${predicted_price:.2f}")
    print(f"   Expected Change: ${price_change:+.2f} ({price_change_percent:+.1f}%)")
    print(f"   Likely Range: ${predicted_low:.2f} to ${predicted_high:.2f}")

    if price_change_percent > 0:
        print(f"   → This suggests the stock might go UP by about {abs(price_change_percent):.1f}%")
    else:
        print(f"   → This suggests the stock might go DOWN by about {abs(price_change_percent):.1f}%")

    # Accuracy info
    if accuracy:
        print(f"\n📊 HOW ACCURATE ARE OUR PREDICTIONS?")
        print("   Based on testing our methods on past data:")
        for period, metrics in accuracy.items():
            days = period.replace('_days', '')
            print(f"   {days}-day predictions: Usually off by about {metrics['error_percentage']:.1f}%")

    # Recommendation
    recommendation = create_simple_recommendation(results)
    print(f"\n💡 RECOMMENDATION:")
    print(f"   {recommendation['recommendation']}")
    print(f"   Why: {recommendation['reasoning']}")
    print(f"   Confidence Level: {recommendation['confidence']}")
    print(
        f"   Positive signals: {recommendation['positive_signals']}, Negative signals: {recommendation['negative_signals']}")

    # Risk warning
    uncertainty = recommendation['uncertainty_percent']
    print(f"\n⚠️  IMPORTANT NOTES:")
    if uncertainty > 30:
        print("   🔴 HIGH UNCERTAINTY: Price could vary a lot from prediction")
    elif uncertainty > 20:
        print("   🟡 MEDIUM UNCERTAINTY: Some variation from prediction expected")
    else:
        print("   🟢 LOW UNCERTAINTY: Price likely to be close to prediction")

    print("   📝 This is just a prediction based on past patterns")
    print("   📝 Stock prices can change due to unexpected news or events")
    print("   📝 Never invest money you can't afford to lose")
    print("   📝 Consider consulting a financial advisor for major decisions")

    print("=" * 80)


def display_portfolio_analysis(results: Dict, buy_price: float, units: int):
    """Show what this means for your specific investment"""

    forecast = results['forecast']
    predicted_price = forecast.iloc[-1]['yhat']
    predicted_low = forecast.iloc[-1]['yhat_lower']
    predicted_high = forecast.iloc[-1]['yhat_upper']

    # Calculate portfolio values
    initial_investment = buy_price * units
    predicted_value = predicted_price * units
    worst_case_value = predicted_low * units
    best_case_value = predicted_high * units

    # Calculate gains/losses
    predicted_gain = predicted_value - initial_investment
    predicted_gain_percent = (predicted_gain / initial_investment) * 100
    worst_case_gain = worst_case_value - initial_investment
    best_case_gain = best_case_value - initial_investment

    print(f"\n💼 YOUR PORTFOLIO ANALYSIS:")
    print(f"   Your Position: {units} shares at ${buy_price:.2f} each")
    print(f"   Total Investment: ${initial_investment:,.2f}")
    print("")
    print(f"   📈 Expected Value: ${predicted_value:,.2f}")
    print(f"      Expected Gain/Loss: ${predicted_gain:+,.2f} ({predicted_gain_percent:+.1f}%)")
    print("")
    print(f"   📊 Possible Range:")
    print(f"      Best Case: ${best_case_value:,.2f} (gain: ${best_case_gain:+,.2f})")
    print(f"      Worst Case: ${worst_case_value:,.2f} (loss: ${worst_case_gain:+,.2f})")

    # Simple advice
    if predicted_gain_percent > 10:
        print(f"\n   💰 Good news! Your investment might grow nicely")
    elif predicted_gain_percent > 0:
        print(f"\n   📈 Modest gains expected for your investment")
    elif predicted_gain_percent > -10:
        print(f"\n   📉 Small loss possible - monitor closely")
    else:
        print(f"\n   ⚠️  Significant loss possible - consider your options carefully")


def create_simple_plots(results: Dict, ticker: str):
    """Create easy-to-understand charts"""
    try:
        stock_data = results['stock_data']
        forecast = results['forecast']

        # Calculate moving averages for plotting
        stock_data_with_avg = stock_data.copy()
        stock_data_with_avg['20_day_avg'] = stock_data['Close'].rolling(window=20).mean()
        stock_data_with_avg['50_day_avg'] = stock_data['Close'].rolling(window=50).mean()

        # Create 2 charts
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Chart 1: Price history and prediction
        ax1.plot(stock_data.index, stock_data['Close'], 'b-', label='Actual Price', linewidth=2)
        ax1.plot(stock_data_with_avg.index, stock_data_with_avg['20_day_avg'], 'orange',
                 linestyle='--', label='20-day Average', alpha=0.7)
        ax1.plot(stock_data_with_avg.index, stock_data_with_avg['50_day_avg'], 'red',
                 linestyle='--', label='50-day Average', alpha=0.7)

        # Add prediction
        ax1.plot(forecast['ds'], forecast['yhat'], 'green', linewidth=3, label='Price Prediction')
        ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                         color='green', alpha=0.2, label='Prediction Range')

        ax1.set_title(f'{ticker.upper()} - Price History and Prediction', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Chart 2: Trading volume
        ax2.bar(stock_data.index, stock_data['Volume'], alpha=0.6, color='lightblue',
                label='Daily Volume')
        volume_avg = stock_data['Volume'].rolling(window=20).mean()
        ax2.plot(stock_data.index, volume_avg, 'red', linewidth=2, label='20-day Average Volume')

        ax2.set_title('Trading Volume (How Many Shares Were Traded)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Shares', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n📊 CHART EXPLANATION:")
        print("   📈 Blue line = actual stock price over time")
        print("   🟠 Orange line = 20-day average (recent trend)")
        print("   🔴 Red line = 50-day average (longer-term trend)")
        print("   🟢 Green line = our prediction for the next 3 months")
        print("   🟢 Green area = range where price will likely be")
        print("   📊 Volume shows how much trading activity there was each day")

    except Exception as e:
        print(f"Could not create charts: {e}")


def main():
    """Main function with original usage format"""
    parser = argparse.ArgumentParser(description="Simple Stock Analysis Tool")
    parser.add_argument("ticker", help="Stock symbol (like AAPL, TSLA, 5258.KL)")
    parser.add_argument("--buy", type=float, help="Price you bought the stock at")
    parser.add_argument("--units", type=int, help="How many shares/units you own")

    args = parser.parse_args()

    # Validate portfolio inputs - both must be provided together
    if (args.buy is not None and args.units is None) or \
            (args.buy is None and args.units is not None):
        parser.error("Arguments --buy and --units must be used together")

    print("🚀 Simple Stock Analysis Tool")
    print("This tool analyzes stocks and explains everything in simple terms\n")

    # Create analyzer and run analysis
    analyzer = SimpleStockAnalyzer()
    results = analyzer.analyze_stock(args.ticker)

    if "error" in results:
        print(f"❌ {results['error']}")
        return

    # Show results
    display_simple_results(results, args.ticker)

    # Show portfolio analysis if requested
    if args.buy is not None and args.units is not None:
        display_portfolio_analysis(results, args.buy, args.units)

    # Show charts based on configuration
    if SHOW_PLOTS:
        print(f"\n📊 Creating easy-to-understand charts...")
        create_simple_plots(results, args.ticker)


if __name__ == "__main__":
    main()