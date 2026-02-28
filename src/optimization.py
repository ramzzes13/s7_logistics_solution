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
        """
        if 'lead_time_days' not in df_orders.columns:
            df_orders['lead_time_days'] = (pd.to_datetime(df_orders['delivery_date']) - pd.to_datetime(df_orders['order_date'])).dt.days
            
        mean_lt = df_orders['lead_time_days'].mean()
        std_lt = df_orders['lead_time_days'].std()
        
        # Handle cases with 1 or 0 orders
        if pd.isna(mean_lt):
            mean_lt = 7 # Default fallback
            std_lt = 0
            
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

    def calculate_eoq(self, annual_demand, ordering_cost, holding_cost_per_unit):
        """
        Calculates Economic Order Quantity.
        EOQ = sqrt(2 * D * S / H)
        """
        if holding_cost_per_unit <= 0:
            return annual_demand / 12 # Fallback to monthly demand
            
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        return eoq

    def generate_procurement_plan(self, category, demand_forecast, price_forecast, lead_time_mean, 
                                  current_inventory=0, safety_stock=0, reorder_point=0, 
                                  batch_size=None, min_order_qty=0, avg_daily_demand=0):
        """
        Simulates inventory and generates orders.
        demand_forecast: DataFrame with 'ds' and 'yhat' (daily demand).
        price_forecast: DataFrame with 'ds' and 'yhat' (daily price).
        """
        self.logger.info(f"Generating procurement plan for {category}...")
        
        plan = []
        # Initial inventory assumption
        inventory = reorder_point 
        
        pending_orders = [] # List of (arrival_date, qty)
        
        # Merge forecasts to align dates
        # Use left join on demand forecast to ensure we cover the demand period
        df_merged = pd.merge(demand_forecast, price_forecast, on='ds', how='left', suffixes=('_demand', '_price'))
        df_merged = df_merged.sort_values('ds')
        
        # Fill missing prices with ffill or mean
        if df_merged['yhat_price'].isnull().any():
            self.logger.warning(f"Missing price forecasts for {category}. Filling with last known or mean.")
            df_merged['yhat_price'] = df_merged['yhat_price'].ffill().bfill()
            if df_merged['yhat_price'].isnull().all():
                 df_merged['yhat_price'] = 1.0 # Absolute fallback
        
        # We simulate day by day
        for idx, row in df_merged.iterrows():
            current_date = row['ds']
            daily_demand = max(0, row['yhat_demand'])
            unit_price = max(0, row['yhat_price']) # Ensure positive price
            
            # Check for arriving orders
            arrived_qty = sum([o[1] for o in pending_orders if o[0] <= current_date])
            # Remove arrived orders
            pending_orders = [o for o in pending_orders if o[0] > current_date]
            
            inventory += arrived_qty
            inventory -= daily_demand
            
            # Check reorder point
            # Calculate inventory position (Inventory + On Order)
            inventory_position = inventory + sum([o[1] for o in pending_orders])
            
            if inventory_position <= reorder_point:
                # Place order
                # Target level logic: Order up to cover X days (e.g., 30 days) + Safety Stock
                days_coverage = 30 # Default monthly replenishment logic
                target_level = reorder_point + (avg_daily_demand * days_coverage)
                
                if target_level <= reorder_point:
                     target_level = reorder_point + (avg_daily_demand * 30)
                
                order_qty = target_level - inventory_position
                
                if batch_size:
                    # Round to batch size
                    batches = np.ceil(order_qty / batch_size)
                    order_qty = batches * batch_size
                
                if order_qty < min_order_qty:
                    order_qty = min_order_qty
                
                # Round to nearest integer
                order_qty = int(np.ceil(order_qty))
                
                if order_qty > 0:
                    arrival_date = current_date + pd.Timedelta(days=int(lead_time_mean))
                    pending_orders.append((arrival_date, order_qty))
                    
                    amount = order_qty * unit_price
                    
                    plan.append({
                        'product_category': category,
                        'order_date': current_date,
                        'amount': amount,
                        'qty': order_qty
                    })
        
        return pd.DataFrame(plan)
