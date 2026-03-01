import pandas as pd
import numpy as np
import logging

class InventoryOptimizer:
    def __init__(self, service_level=0.95):
        self.service_level = service_level
        self.z_score = 1.645 # Approx for 95%
        if service_level == 0.99:
            self.z_score = 2.33
        elif service_level == 0.90:
            self.z_score = 1.28
            
        self.logger = logging.getLogger(__name__)

    def calculate_lead_time_params(self, df_orders):
        """
        Calculates mean and std of lead time (in days).
        Returns None, None if not enough data.
        """
        if df_orders is None or df_orders.empty:
            return None, None

        if 'lead_time_days' not in df_orders.columns:
            df_orders = df_orders.copy()
            df_orders['lead_time_days'] = (pd.to_datetime(df_orders['delivery_date']) - pd.to_datetime(df_orders['order_date'])).dt.days
            
        # Drop NaNs
        lt_data = df_orders['lead_time_days'].dropna()
        if len(lt_data) == 0:
            return None, None
            
        mean_lt = lt_data.mean()
        std_lt = lt_data.std()
        
        if pd.isna(std_lt):
            std_lt = 0
            
        return mean_lt, std_lt

    def calculate_safety_stock(self, demand_std, lead_time_mean):
        """
        Calculates Safety Stock.
        SS = Z * sigma_demand * sqrt(lead_time)
        Assumes demand_std is per day and lead_time is in days.
        """
        ss = self.z_score * demand_std * np.sqrt(lead_time_mean)
        return ss

    def calculate_reorder_point(self, demand_mean, lead_time_mean, safety_stock):
        """
        Calculates Reorder Point.
        ROP = Avg_demand * Lead_time + SS
        """
        rop = demand_mean * lead_time_mean + safety_stock
        return rop

    def generate_procurement_plan(self, product_id, category, demand_forecast, price_forecast, lead_time_mean, 
                                  current_inventory=0, safety_stock=0, reorder_point=0, 
                                  batch_size=None, min_order_qty=0, avg_daily_demand=0, freq='D'):
        """
        Simulates inventory and generates orders for a specific product.
        """
        # self.logger.info(f"Generating plan for Product {product_id} (Cat {category}, Freq: {freq})...")
        
        # Convert aggregated forecast to daily for simulation if needed
        if freq != 'D':
            df_daily = demand_forecast.set_index('ds').resample('D').asfreq()
            
            # Forward fill the rate, but divide by days in period
            if freq == 'W':
                divider = 7
            elif freq == 'MS':
                divider = 30 # Approx
            else:
                divider = 1
                
            # Fill with previous known rate / divider
            df_daily['yhat'] = demand_forecast.set_index('ds').reindex(df_daily.index).ffill()['yhat'] / divider
            df_daily = df_daily.reset_index()
            demand_forecast = df_daily
        
        plan = []
        inventory = reorder_point 
        
        pending_orders = [] # List of (arrival_date, qty)
        
        # Merge forecasts
        df_merged = pd.merge(demand_forecast, price_forecast, on='ds', how='left', suffixes=('_demand', '_price'))
        df_merged = df_merged.sort_values('ds')
        
        # Fill missing prices
        if df_merged['yhat_price'].isnull().any():
            df_merged['yhat_price'] = df_merged['yhat_price'].ffill().bfill()
            if df_merged['yhat_price'].isnull().all():
                 df_merged['yhat_price'] = 1.0 # Default if absolutely no price info
        
        # Simulation
        for idx, row in df_merged.iterrows():
            current_date = row['ds']
            daily_demand = max(0, row['yhat_demand'])
            unit_price = max(0, row['yhat_price'])
            
            # Check for arriving orders
            arrived_qty = sum([o[1] for o in pending_orders if o[0] <= current_date])
            pending_orders = [o for o in pending_orders if o[0] > current_date]
            
            inventory += arrived_qty
            inventory -= daily_demand
            
            # Check reorder point
            inventory_position = inventory + sum([o[1] for o in pending_orders])
            
            if inventory_position <= reorder_point:
                # Order logic
                days_coverage = 30 
                target_level = reorder_point + (avg_daily_demand * days_coverage)
                
                if target_level <= reorder_point:
                     target_level = reorder_point + (avg_daily_demand * 30)
                
                order_qty = target_level - inventory_position
                
                # Apply Batch Size
                if batch_size:
                    batches = np.ceil(order_qty / batch_size)
                    order_qty = batches * batch_size
                
                if order_qty < min_order_qty:
                    order_qty = min_order_qty
                
                order_qty = int(np.ceil(order_qty))
                
                if order_qty > 0:
                    arrival_date = current_date + pd.Timedelta(days=int(lead_time_mean))
                    pending_orders.append((arrival_date, order_qty))
                    
                    amount = order_qty * unit_price
                    
                    plan.append({
                        'product_id': product_id,
                        'product_category': category,
                        'order_date': current_date,
                        'amount': amount,
                        'qty': order_qty
                    })
        
        return pd.DataFrame(plan)
