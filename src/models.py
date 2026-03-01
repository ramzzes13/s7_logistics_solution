import numpy as np
import pandas as pd
from prophet import Prophet
import logging

class TSBModel:
    """
    Custom implementation of Teunter-Syntetos-Babai (TSB) method for intermittent demand.
    Does not require statsforecast or compilation.
    """
    def __init__(self, alpha_d=0.2, alpha_p=0.2):
        self.alpha_d = alpha_d
        self.alpha_p = alpha_p
        self.demand = None
        self.probability = None
        self.last_forecast = None

    def fit(self, y):
        """
        y: numpy array of demand (including zeros)
        """
        n = len(y)
        # Initialize
        p = np.zeros(n) # Probability of demand
        z = np.zeros(n) # Demand size (when demand occurs)
        
        # Initial values (heuristic)
        # Find first non-zero
        first_nonzero_idx = np.nonzero(y)[0]
        if len(first_nonzero_idx) == 0:
            self.demand = 0
            self.probability = 0
            return self
            
        start = first_nonzero_idx[0]
        p[start] = 1.0 / (1 + start) if start > 0 else 0.5
        z[start] = y[start]
        
        for t in range(start + 1, n):
            if y[t] > 0:
                p[t] = p[t-1] + self.alpha_p * (1 - p[t-1])
                z[t] = z[t-1] + self.alpha_d * (y[t] - z[t-1])
            else:
                p[t] = p[t-1] - self.alpha_p * p[t-1]
                z[t] = z[t-1] # Demand size estimate doesn't change when no demand
                
        self.probability = p[-1]
        self.demand = z[-1]
        self.last_forecast = self.probability * self.demand
        return self

    def predict(self, h):
        """
        Returns constant forecast for h periods
        """
        return np.full(h, self.last_forecast)

class ProphetWrapper:
    """
    Wrapper for Facebook Prophet to handle frequency and aggregation.
    """
    def __init__(self, freq='D'):
        self.freq = freq
        self.model = None
        self.logger = logging.getLogger('ProphetWrapper')

    def fit_predict(self, df, horizon_days=365):
        """
        df: DataFrame with 'ds' and 'y'
        horizon_days: prediction horizon in days
        """
        # Configure Prophet based on frequency
        weekly_seasonality = True if self.freq in ['D', 'W'] else False
        daily_seasonality = False
        yearly_seasonality = True
        
        self.model = Prophet(
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            yearly_seasonality=yearly_seasonality
        )
        
        try:
            self.model.fit(df)
            
            # Determine periods based on frequency
            if self.freq == 'W':
                periods = int(np.ceil(horizon_days / 7))
            elif self.freq == 'MS':
                periods = int(np.ceil(horizon_days / 30))
            else:
                periods = horizon_days
                
            future = self.model.make_future_dataframe(periods=periods, freq=self.freq)
            forecast = self.model.predict(future)
            
            # Clip negative values
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            
            return forecast[['ds', 'yhat']]
        except Exception as e:
            self.logger.error(f"Prophet training failed: {e}")
            return None
