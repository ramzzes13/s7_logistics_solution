import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
CONSUMPTION_FILE = "/Users/chiefboss/Desktop/s7_case/final_consumtion_train.csv"
ORDERS_FILE = "/Users/chiefboss/Desktop/s7_case/final_orders_train.csv"
PRODUCT_STATS_FILE = "/Users/chiefboss/Desktop/s7_case/product_stats.xlsx"

def load_data():
    print("Loading data...")
    try:
        df_consumption = pd.read_csv(CONSUMPTION_FILE)
        print(f"Consumption data loaded: {df_consumption.shape}")
    except Exception as e:
        print(f"Error loading consumption data: {e}")
        df_consumption = None

    try:
        df_orders = pd.read_csv(ORDERS_FILE)
        print(f"Orders data loaded: {df_orders.shape}")
    except Exception as e:
        print(f"Error loading orders data: {e}")
        df_orders = None

    try:
        df_stats = pd.read_excel(PRODUCT_STATS_FILE)
        print(f"Product stats loaded: {df_stats.shape}")
    except Exception as e:
        print(f"Error loading product stats: {e}")
        df_stats = None
        
    return df_consumption, df_orders, df_stats

def analyze_consumption(df):
    if df is None:
        return
    
    print("\n--- Consumption Data Analysis ---")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Check if 'consumtion_date' is datetime
    if 'consumtion_date' in df.columns:
        df['consumtion_date'] = pd.to_datetime(df['consumtion_date'])
        print(f"\nDate range: {df['consumtion_date'].min()} to {df['consumtion_date'].max()}")
    
    if 'product_category' in df.columns:
        print("\nConsumption by Category:")
        stats = df.groupby('product_category')['qty'].agg(['count', 'sum', 'mean', 'std']).sort_values('sum', ascending=False)
        print(stats)
        
        # Calculate volatility (coefficient of variation)
        stats['cv'] = stats['std'] / stats['mean']
        print("\nVolatility (CV) by Category:")
        print(stats[['cv']].sort_values('cv', ascending=False))

def analyze_orders(df):
    if df is None:
        return

    print("\n--- Orders Data Analysis ---")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check dates
    for col in ['order_date', 'delivery_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    if 'order_date' in df.columns and 'delivery_date' in df.columns:
        df['lead_time'] = (df['delivery_date'] - df['order_date']).dt.days
        print("\nLead Time Statistics (days):")
        print(df['lead_time'].describe())
        
        if 'product_category' in df.columns:
            print("\nLead Time by Category:")
            print(df.groupby('product_category')['lead_time'].agg(['mean', 'median', 'std', 'min', 'max']))

    if 'amount' in df.columns and 'qty' in df.columns:
        # Avoid division by zero
        df['unit_price'] = df['amount'] / df['qty'].replace(0, np.nan)
        print("\nUnit Price Statistics:")
        print(df['unit_price'].describe())
        
        if 'product_category' in df.columns:
             print("\nUnit Price by Category:")
             print(df.groupby('product_category')['unit_price'].agg(['mean', 'std', 'min', 'max']))

def main():
    df_consumption, df_orders, df_stats = load_data()
    
    analyze_consumption(df_consumption)
    analyze_orders(df_orders)
    
    if df_stats is not None:
        print("\n--- Product Stats ---")
        print(df_stats.head())

if __name__ == "__main__":
    main()
