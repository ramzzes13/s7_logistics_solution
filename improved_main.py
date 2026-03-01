import sys
import os
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from models import TSBModel, ProphetWrapper
from forecasting import PriceForecaster
from optimization import InventoryOptimizer
import utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CONSUMPTION_FILE = "/Users/chiefboss/Desktop/s7_case/final_consumtion_train.csv"
ORDERS_FILE = "/Users/chiefboss/Desktop/s7_case/final_orders_train.csv"
PRODUCT_STATS_FILE = "/Users/chiefboss/Desktop/s7_case/product_stats.xlsx"
OUTPUT_DIR = "s7_logistics_solution/output_improved"
OUTPUT_FILE = "s7_logistics_solution/output_improved/procurement_plan.csv"

def fill_time_series(df, freq='MS'):
    """
    Fills missing dates with 0 demand for intermittent series.
    """
    if df.empty:
        return df
        
    start_date = df['consumtion_date'].min()
    end_date = df['consumtion_date'].max()
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # First, aggregate duplicates on the same day if any
    df_agg = df.groupby(['product_id', 'consumtion_date'])['qty'].sum().reset_index()
    
    filled_dfs = []
    for pid, group in df_agg.groupby('product_id'):
        group = group.set_index('consumtion_date')
        # Reindex to fill missing dates with 0
        group = group.reindex(all_dates, fill_value=0)
        group['product_id'] = pid
        group['product_category'] = df['product_category'].iloc[0]
        group['qty'] = group['qty'].fillna(0)
        filled_dfs.append(group.reset_index().rename(columns={'index': 'consumtion_date'}))
        
    if filled_dfs:
        return pd.concat(filled_dfs, ignore_index=True)
    return df

def main():
    logger.info("Starting PRODUCT-LEVEL S7 Logistics Optimization...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    data_loader = DataLoader(CONSUMPTION_FILE, ORDERS_FILE, PRODUCT_STATS_FILE)
    df_consumption, df_orders, df_stats = data_loader.load_data()
    
    df_consumption = data_loader.preprocess_consumption(df_consumption)
    df_orders = data_loader.preprocess_orders(df_orders)
    
    price_forecaster = PriceForecaster()
    optimizer = InventoryOptimizer(service_level=0.95)
    
    categories = df_consumption['product_category'].unique()
    all_plans = []
    
    # Exclusions
    EXCLUDED_CATEGORIES = [1]
    EXCLUDED_PRODUCTS = [49, 43, 55, 41, 40, 52, 57, 53, 62, 45, 46, 47, 42, 21]

    for category in categories:
        if category in EXCLUDED_CATEGORIES:
            logger.info(f"Skipping excluded Category: {category}")
            continue

        logger.info(f"=== Processing Category: {category} ===")
        
        # Category Level Data
        cat_consumption_all = df_consumption[df_consumption['product_category'] == category].copy()
        cat_orders_all = df_orders[df_orders['product_category'] == category].copy()
        
        # Calculate Category Fallback Params (in case product has no history)
        cat_mean_lt, cat_std_lt = optimizer.calculate_lead_time_params(cat_orders_all)
        if cat_mean_lt is None:
            cat_mean_lt = 30 # Default fallback
            
        # Category Price Forecast (Fallback)
        cat_price_forecast = price_forecaster.forecast_price(cat_orders_all, horizon_days=365)
        
        # Iterate over Products
        products = cat_consumption_all['product_id'].unique()
        logger.info(f"Found {len(products)} products in category {category}")
        
        for product_id in products:
            if product_id in EXCLUDED_PRODUCTS:
                logger.info(f"Skipping excluded Product: {product_id}")
                continue

            # Product Level Data
            prod_consumption = cat_consumption_all[cat_consumption_all['product_id'] == product_id].copy()
            prod_orders = cat_orders_all[cat_orders_all['product_id'] == product_id].copy()
            
            # --- 1. Demand Forecast ---
            demand_forecast = None
            freq = 'D'
            
            if category in [0, 1]:
                # Intermittent -> Monthly + TSB
                freq = 'MS'
                prod_filled = fill_time_series(prod_consumption, freq=freq)
                # Aggregate to ensure unique dates
                prod_agg = prod_filled.groupby('consumtion_date')['qty'].sum().reset_index()
                demand_values = prod_agg['qty'].values
                
                if len(demand_values) < 2:
                    logger.warning(f"Not enough data for Product {product_id}. Skipping.")
                    continue

                tsb = TSBModel(alpha_d=0.2, alpha_p=0.2)
                tsb.fit(demand_values)
                
                horizon = 12
                forecast_values = tsb.predict(horizon)
                future_dates = pd.date_range(start=prod_agg['consumtion_date'].max() + pd.Timedelta(days=1), periods=horizon, freq='MS')
                demand_forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
                
            elif category == 3:
                # Mass -> Weekly + Prophet
                freq = 'W'
                prod_agg = prod_consumption.groupby([pd.Grouper(key='consumtion_date', freq='W')])['qty'].sum().reset_index()
                prod_agg = prod_agg.rename(columns={'consumtion_date': 'ds', 'qty': 'y'})
                
                if len(prod_agg) < 5:
                     # Fallback to monthly if not enough weekly points
                     freq = 'MS'
                     prod_agg = prod_consumption.groupby([pd.Grouper(key='consumtion_date', freq='MS')])['qty'].sum().reset_index()
                     prod_agg = prod_agg.rename(columns={'consumtion_date': 'ds', 'qty': 'y'})
                
                prophet = ProphetWrapper(freq=freq)
                demand_forecast = prophet.fit_predict(prod_agg, horizon_days=365)
                
            else: # Categories 2, 4
                # Regular -> Monthly + Prophet
                freq = 'MS'
                prod_agg = prod_consumption.groupby([pd.Grouper(key='consumtion_date', freq='MS')])['qty'].sum().reset_index()
                prod_agg = prod_agg.rename(columns={'consumtion_date': 'ds', 'qty': 'y'})
                
                if len(prod_agg) < 2:
                    logger.warning(f"Not enough data for Product {product_id}. Skipping.")
                    continue
                    
                prophet = ProphetWrapper(freq=freq)
                demand_forecast = prophet.fit_predict(prod_agg, horizon_days=365)

            if demand_forecast is None or demand_forecast.empty:
                logger.error(f"Failed to forecast for Product {product_id}")
                continue

            # --- 2. Lead Time ---
            mean_lt, std_lt = optimizer.calculate_lead_time_params(prod_orders)
            if mean_lt is None:
                # Use Category Average
                mean_lt = cat_mean_lt
                
            # --- 3. Price Forecast ---
            # Try to forecast price for product, else use category forecast
            if len(prod_orders) > 5:
                price_forecast = price_forecaster.forecast_price(prod_orders, horizon_days=365)
            else:
                price_forecast = cat_price_forecast

            # --- 4. Inventory Parameters ---
            total_days = (prod_consumption['consumtion_date'].max() - prod_consumption['consumtion_date'].min()).days
            if total_days > 0:
                avg_daily_rate = prod_consumption['qty'].sum() / total_days
            else:
                avg_daily_rate = prod_consumption['qty'].mean() # Fallback
                
            if avg_daily_rate <= 0: avg_daily_rate = 0.1 # Minimum to avoid div by zero issues
            
            # Variability
            if category in [0, 1]:
                 demand_std = avg_daily_rate * 2 
            else:
                 demand_std = avg_daily_rate * 0.5
                 
            ss = optimizer.calculate_safety_stock(demand_std, mean_lt)
            rop = optimizer.calculate_reorder_point(avg_daily_rate, mean_lt, ss)
            
            # Filter Forecast Period
            start_date = '2025-09-01'
            end_date = '2026-09-01'
            mask = (demand_forecast['ds'] >= start_date) & (demand_forecast['ds'] <= end_date)
            period_demand = demand_forecast.loc[mask].copy()
            
            if period_demand.empty:
                 continue

            # Generate Plan
            plan_df = optimizer.generate_procurement_plan(
                product_id,
                category,
                period_demand,
                price_forecast,
                mean_lt,
                current_inventory=rop,
                safety_stock=ss,
                reorder_point=rop,
                min_order_qty=1, # Can order 1 item
                avg_daily_demand=avg_daily_rate,
                freq=freq
            )
            
            if not plan_df.empty:
                all_plans.append(plan_df)

    # Save Final Plan
    if all_plans:
        final_plan = pd.concat(all_plans, ignore_index=True)
        # Reorder columns
        final_plan = final_plan[['product_category', 'product_id', 'order_date', 'amount', 'qty']]
        final_plan = final_plan.sort_values(['product_category', 'order_date'])
        
        utils.save_plan(final_plan, OUTPUT_FILE)
        logger.info(f"Detailed Product Procurement Plan saved to {OUTPUT_FILE}")
        print(final_plan.head())
    else:
        logger.warning("No plan generated.")

if __name__ == "__main__":
    main()
