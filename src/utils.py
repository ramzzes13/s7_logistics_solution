import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_forecasts(history, forecast, category, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(history['ds'], history['y'], label='Historical')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    plt.title(f'Demand Forecast for {category}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'forecast_{category}.png'))
    plt.close()

def plot_lead_time_dist(lead_times, category, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(lead_times, kde=True)
    plt.title(f'Lead Time Distribution for {category}')
    plt.xlabel('Days')
    plt.savefig(os.path.join(output_dir, f'lead_time_{category}.png'))
    plt.close()

def save_plan(plan_df, output_path):
    plan_df.to_csv(output_path, index=False)
