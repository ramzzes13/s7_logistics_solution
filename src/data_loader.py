import pandas as pd
import numpy as np
import logging

class DataLoader:
    def __init__(self, consumption_path, orders_path, stats_path):
        self.consumption_path = consumption_path
        self.orders_path = orders_path
        self.stats_path = stats_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.logger.info("Loading data...")
        try:
            df_consumption = pd.read_csv(self.consumption_path)
            df_orders = pd.read_csv(self.orders_path)
            df_stats = pd.read_excel(self.stats_path)
            
            return df_consumption, df_orders, df_stats
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def preprocess_consumption(self, df):
        self.logger.info("Preprocessing consumption data...")
        df = df.copy()
        # Convert date
        df['consumtion_date'] = pd.to_datetime(df['consumtion_date'])
        
        # Handle duplicates/nulls
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicates in consumption data. Dropping them.")
            df = df.drop_duplicates()
            
        # Sort by date
        df = df.sort_values('consumtion_date')
        
        return df

    def preprocess_orders(self, df):
        self.logger.info("Preprocessing orders data...")
        df = df.copy()
        # Convert dates
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['delivery_date'] = pd.to_datetime(df['delivery_date'])
        
        # Calculate lead time
        df['lead_time_days'] = (df['delivery_date'] - df['order_date']).dt.days
        
        # Calculate unit price
        # Check for zero qty to avoid division by zero
        df['unit_price'] = df.apply(lambda row: row['amount'] / row['qty'] if row['qty'] > 0 else 0, axis=1)
        
        return df

    def aggregate_consumption(self, df, freq='W'):
        """
        Aggregates consumption by category and frequency.
        freq: 'D' for daily, 'W' for weekly, 'M' for monthly.
        """
        self.logger.info(f"Aggregating consumption by category with frequency {freq}...")
        
        # Group by category and resample
        # Set index to date for resampling
        df_agg = df.set_index('consumtion_date').groupby('product_category')['qty'].resample(freq).sum().reset_index()
        
        return df_agg

    def get_category_stats(self, df_consumption):
        """Calculates basic stats per category."""
        stats = df_consumption.groupby('product_category')['qty'].agg(['count', 'mean', 'std', 'sum'])
        stats['cv'] = stats['std'] / stats['mean']  # Coefficient of variation
        return stats
