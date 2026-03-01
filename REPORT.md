# S7 Airlines Procurement Optimization Model (Enhanced)

## Overview
This solution provides a high-precision procurement plan for S7 Airlines for the period **01.09.2025 – 01.09.2026**. Unlike the initial baseline, this enhanced version performs forecasting and inventory planning at the **individual product level (SKU)** rather than the aggregate category level, ensuring specific needs for every component are met.

## Methodology

### 1. Hybrid Forecasting Strategy
We implemented a dynamic model selection pipeline based on the demand patterns of each category:

| Category | Demand Type | Aggregation | Model | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **0, 1** | **Intermittent / Rare** | Monthly (`MS`) | **TSB (Teunter-Syntetos-Babai)** | Best suited for sparse data with many zeros. Estimates demand probability and size separately. |
| **3** | **Mass / High Volatility** | Weekly (`W`) | **Prophet** | Captures complex seasonality (weekly/yearly) and trend shifts better than simple smoothing. |
| **2, 4** | **Regular / Stable** | Monthly (`MS`) | **Prophet** | Robust forecasting for standard consumption patterns. |

*Note: The TSB model was implemented from scratch using NumPy to avoid heavy dependencies like `statsforecast` requiring Fortran compilers.*

### 2. Product-Level Granularity
- **Demand:** Forecasts are generated for each unique `product_id`.
- **Lead Time:** Calculated based on the specific order history of the product.
- **Price:** Forecasted per product using historical trends.
- **Fallback Mechanism:** If a specific product lacks sufficient history (e.g., new or extremely rare items), the system automatically falls back to **Category Averages** for Lead Time and Price to ensure continuity.

### 3. Inventory Optimization
The model simulates inventory levels on a **daily basis** to determine exactly when to order:
- **Safety Stock (SS):** Dynamic calculation based on demand variability ($\sigma_{demand}$) and Lead Time ($\sqrt{LT}$).
- **Reorder Point (ROP):** Triggers an order when `Inventory Position <= ROP`.
- **Order Quantity:** Calculated to cover the review period plus safety stock, respecting minimum order quantities.

### 4. Data Processing
- **Temporal Alignment:** Weekly and Monthly forecasts are interpolated to daily rates to interact seamlessly with the daily inventory simulation.
- **Gap Filling:** For intermittent series, missing dates are rigorously filled with zeros to prevent model bias.

## Key Files
- `improved_main.py`: The orchestration script running the full pipeline.
- `src/models.py`: Custom implementations of TSB and Prophet wrappers.
- `src/optimization.py`: Inventory logic supporting variable forecast frequencies.
- `output_improved/procurement_plan.csv`: The final detailed schedule containing `product_id`, `order_date`, `amount`, and `qty`.

## Results
The resulting plan minimizes stockouts for high-velocity items (Category 3) by checking status weekly, while maintaining efficient buffer stocks for expensive, slow-moving parts (Category 0/1).
