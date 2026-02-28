# S7 Airlines Procurement Optimization Model

## Overview
This solution provides a comprehensive procurement plan for S7 Airlines for the period 01.09.2025 – 01.09.2026. The model analyzes historical consumption and order data to optimize inventory levels for 5 distinct product categories.

## Methodology

### 1. Data Analysis (EDA)
- **Consumption Data**: Aggregated by product category to understand demand patterns, volatility, and seasonality.
- **Orders Data**: Used to calculate historical Lead Times and unit prices.
- **Key Metrics**:
  - **Demand Volatility (CV)**: Used to determine inventory strategy.
  - **Lead Time Variability**: Used for Safety Stock calculation.

### 2. Demand Forecasting
- **Model**: **Prophet** (Facebook)
- **Rationale**: Prophet is robust to missing data and handles seasonality (weekly/yearly) well, which is crucial for logistics. It also provides uncertainty intervals.
- **Aggregation**: Daily demand aggregation was used to capture granular patterns.
- **Horizon**: 12 months (365 days).

### 3. Price Forecasting
- **Strategy**:
  - **Low Volatility (CV < 0.1)**: Used historical average price.
  - **High Volatility**: Used Prophet to forecast price trends, ensuring non-negative prices.

### 4. Inventory Optimization
- **Policy**: Continuous Review (s, Q) / Periodic Review hybrid simulation.
- **Parameters**:
  - **Safety Stock (SS)**: Calculated based on service level (95%) and demand/lead time variability.
    - Formula: $SS = Z \times \sigma_{demand} \times \sqrt{LT}$
  - **Reorder Point (ROP)**: Inventory level at which a new order is triggered.
    - Formula: $ROP = \text{Avg Demand} \times \text{Avg LT} + SS$
  - **Order Quantity**: Dynamic calculation based on target inventory coverage (30 days) + Safety Stock.
    - Logic: Order enough to cover expected demand during lead time + review period + safety buffer.

### 5. Procurement Plan Generation
- The simulation runs daily for the planning horizon.
- Orders are placed when projected inventory falls below ROP.
- Output includes `order_date`, `product_category`, `amount` (estimated cost), and `qty`.

## Results Summary by Category

| Category | Strategy | Mean Lead Time (Days) | Safety Stock | ROP | First Order Qty |
|----------|----------|-----------------------|--------------|-----|-----------------|
| **0** | JIT / ROP | 81.8 | 8.4 | 14.3 | ~10 |
| **1** | High Value / Low Stock | 40.8 | 2.6 | 5.0 | ~10 |
| **2** | High Volatility / Buffer | 93.2 | 633.6 | 1037.3 | ~150 |
| **3** | High Volume / Frequent | 44.0 | 6318.5 | 14264.5 | ~5600 |
| **4** | Stable / Periodic | 69.9 | 48.3 | 441.7 | ~170 |

*Note: Quantities are approximate based on initial simulation run.*

## Files
- `procurement_plan.csv`: The detailed procurement schedule.
- `src/`: Source code for the solution (modular Python).
- `output/`: Generated plots for forecasts and lead time distributions.
