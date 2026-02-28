import sys
import os
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from forecasting import DemandForecaster, PriceForecaster
from optimization import InventoryOptimizer
import utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CONSUMPTION_FILE = "/Users/chiefboss/Desktop/s7_case/final_consumtion_train.csv"
ORDERS_FILE = "/Users/chiefboss/Desktop/s7_case/final_orders_train.csv"
PRODUCT_STATS_FILE = "/Users/chiefboss/Desktop/s7_case/product_stats.xlsx"
OUTPUT_DIR = "s7_logistics_solution/output"
OUTPUT_FILE = "s7_logistics_solution/output/procurement_plan.csv"

def main():
    logger.info("Starting S7 Logistics Optimization...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    data_loader = DataLoader(CONSUMPTION_FILE, ORDERS_FILE, PRODUCT_STATS_FILE)
    df_consumption, df_orders, df_stats = data_loader.load_data()
    
    # Preprocess
    df_consumption = data_loader.preprocess_consumption(df_consumption)
    df_orders = data_loader.preprocess_orders(df_orders)
    
    # Get categories
    categories = df_consumption['product_category'].unique()
    logger.info(f"Found {len(categories)} categories: {categories}")
    
    # Initialize forecasters and optimizer
    demand_forecaster = DemandForecaster()
    price_forecaster = PriceForecaster()
    optimizer = InventoryOptimizer(service_level=0.95)
    
    all_plans = []
    
    for category in categories:
        logger.info(f"Processing category: {category}")
        
        # Filter data
        cat_consumption = df_consumption[df_consumption['product_category'] == category].copy()
        cat_orders = df_orders[df_orders['product_category'] == category].copy()
        
        # Aggregate consumption to daily for forecasting (Prophet handles daily well)
        # Or weekly? Let's use daily aggregation, filling missing days with 0?
        # Consumption data might be sparse. Let's aggregate to daily.
        cat_daily = cat_consumption.set_index('consumtion_date').resample('D')['qty'].sum().reset_index()
        cat_daily.columns = ['ds', 'y']
        
        # Forecast Demand (12 months = 365 days)
        # Check if enough data
        if len(cat_daily) < 10:
            logger.warning(f"Not enough data for {category}. Skipping.")
            continue
            
        _, demand_forecast = demand_forecaster.train_predict(cat_daily, horizon_days=365)
        
        if demand_forecast is None:
            logger.error(f"Failed to forecast demand for {category}")
            continue
            
        # Plot forecast
        utils.plot_forecasts(cat_daily, demand_forecast, category, OUTPUT_DIR)
        
        # Forecast Price
        price_forecast = price_forecaster.forecast_price(cat_orders, horizon_days=365)
        
        # Calculate Lead Time & Policy Params
        mean_lt, std_lt = optimizer.calculate_lead_time_params(cat_orders)
        
        # Calculate daily demand stats for SS
        # Use last 90 days stats or full history?
        # Use full history std dev
        daily_demand_mean = cat_daily['y'].mean()
        daily_demand_std = cat_daily['y'].std()
        
        ss = optimizer.calculate_safety_stock(daily_demand_std, mean_lt)
        rop = optimizer.calculate_reorder_point(daily_demand_mean, mean_lt, ss)
        
        logger.info(f"Category {category}: Mean LT={mean_lt:.1f}, SS={ss:.1f}, ROP={rop:.1f}")
        
        # Generate Plan
        # We need the forecast for the period 01.09.2025 - 01.09.2026
        # Filter forecast for this period
        start_date = '2025-09-01'
        end_date = '2026-09-01'
        mask = (demand_forecast['ds'] >= start_date) & (demand_forecast['ds'] <= end_date)
        period_demand = demand_forecast.loc[mask].copy()
        period_price = price_forecast.copy() # Price forecast covers the whole horizon
        
        if period_demand.empty:
            logger.warning(f"No demand forecast for target period for {category}")
            continue
            
        plan_df = optimizer.generate_procurement_plan(
            category, 
            period_demand, 
            period_price, 
            mean_lt, 
            current_inventory=rop, # Assume starting at ROP
            safety_stock=ss, 
            reorder_point=rop,
            min_order_qty=10, # heuristic minimum
            avg_daily_demand=daily_demand_mean
        )
        
        if not plan_df.empty:
            all_plans.append(plan_df)
            
    # Combine all plans
    if all_plans:
        final_plan = pd.concat(all_plans, ignore_index=True)
        # Format output
        # Columns: product_category, order_date, amount, qty
        final_plan = final_plan[['product_category', 'order_date', 'amount', 'qty']]
        # Sort by date
        final_plan = final_plan.sort_values('order_date')
        
        utils.save_plan(final_plan, OUTPUT_FILE)
        logger.info(f"Procurement plan saved to {OUTPUT_FILE}")
        print(final_plan.head())
    else:
        logger.warning("No plan generated.")

if __name__ == "__main__":
    main()
