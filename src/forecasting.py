import pandas as pd
import numpy as np
import logging
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

class DemandForecaster:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train_predict(self, df_history, horizon_days=365):
        """
        df_history: DataFrame with 'ds' (date) and 'y' (value) columns.
        horizon_days: int, number of days to forecast.
        """
        self.logger.info("Training Prophet model...")
        
        # Renaissance of error handling
        if len(df_history) < 2:
            self.logger.warning("Not enough data points for forecasting.")
            return None, None
            
        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_history)
            
            future = model.make_future_dataframe(periods=horizon_days)
            forecast = model.predict(future)
            
            return model, forecast
        except Exception as e:
            self.logger.error(f"Prophet training failed: {e}")
            return None, None

    def evaluate(self, df_history, test_days=90):
        """
        Splits data into train/test and evaluates the model.
        """
        cutoff_date = df_history['ds'].max() - pd.Timedelta(days=test_days)
        train_df = df_history[df_history['ds'] <= cutoff_date]
        test_df = df_history[df_history['ds'] > cutoff_date]
        
        if len(train_df) < 10 or len(test_df) < 1:
            self.logger.warning("Not enough data for evaluation.")
            return {}

        model, forecast = self.train_predict(train_df, horizon_days=test_days)
        
        if forecast is None:
            return {}
            
        # Merge forecast with actuals
        merged = pd.merge(test_df, forecast[['ds', 'yhat']], on='ds', how='inner')
        
        mape = mean_absolute_percentage_error(merged['y'], merged['yhat'])
        rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
        
        return {'MAPE': mape, 'RMSE': rmse}

class PriceForecaster:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def forecast_price(self, df_orders, horizon_days=365):
        """
        Forecasts price based on historical order data.
        df_orders: DataFrame with 'order_date' and 'unit_price'.
        """
        self.logger.info("Forecasting price...")
        
        # Simple approach: Check for trend. If stable, use mean.
        # If significant trend, use linear regression or Prophet.
        
        df = df_orders[['order_date', 'unit_price']].dropna().rename(columns={'order_date': 'ds', 'unit_price': 'y'})
        df = df.sort_values('ds')
        
        if len(df) < 5:
             self.logger.warning("Not enough price data. Using last known price.")
             return pd.DataFrame({'ds': pd.date_range(start=df['ds'].max(), periods=horizon_days, freq='D'), 
                                  'yhat': df['y'].iloc[-1]})

        # Filter out zero prices for stats
        valid_prices = df[df['y'] > 0]['y']
        if valid_prices.empty:
            min_price = 1.0 # Fallback
            cv = 0.0 # Default stable
        else:
            min_price = valid_prices.min()
            cv = valid_prices.std() / valid_prices.mean()
            
        if cv < 0.1: # Low volatility, use mean
            avg_price = df['y'].mean()
            self.logger.info(f"Price is stable (CV={cv:.2f}). Using average price: {avg_price}")
            future_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(days=1), periods=horizon_days, freq='D')
            return pd.DataFrame({'ds': future_dates, 'yhat': avg_price})
        else:
            self.logger.info(f"Price shows volatility (CV={cv:.2f}). Using Prophet for price forecast.")
            model = Prophet(yearly_seasonality=True) # Assuming potential yearly price updates
            model.fit(df)
            future = model.make_future_dataframe(periods=horizon_days)
            forecast = model.predict(future)
            # Ensure no negative prices, clamp to min historical price or reasonable floor
            forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, min_price))
            return forecast[['ds', 'yhat']]
