import pandas as pd
import numpy as np

def calculate_historical_spreads(df):
    """
    Calculates the historical retail premium (spread) and buyback discount
    relative to the Spot IDR Gram price.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'Spot_IDR_Gram', 'Retail_Price', 'Buyback_Price'
        
    Returns:
        dict: A dictionary containing:
            - 'avg_retail_spread': float (e.g., 0.0499 for 4.99%)
            - 'avg_buyback_ratio': float (e.g., 0.952)
            - 'latest_retail_spread': float
            - 'latest_buyback_ratio': float
    """
    # Filter valid rows (no NaNs or zeros)
    valid_df = df[
        (df['Spot_IDR_Gram'] > 0) & 
        (df['Retail_Price'] > 0) & 
        (df['Buyback_Price'] > 0)
    ].copy()
    
    if len(valid_df) == 0:
        # Default fallback values if no valid data
        return {
            'avg_retail_spread': 0.0499,
            'avg_buyback_ratio': 0.952,
            'latest_retail_spread': 0.0499,
            'latest_buyback_ratio': 0.952
        }
    
    # Calculate daily spreads
    # Retail Premium over Spot: (Retail - Spot) / Spot
    valid_df['Retail_Spread'] = (valid_df['Retail_Price'] - valid_df['Spot_IDR_Gram']) / valid_df['Spot_IDR_Gram']
    # Buyback Ratio relative to Retail: Buyback / Retail
    valid_df['Buyback_Ratio'] = valid_df['Buyback_Price'] / valid_df['Retail_Price']
    
    # Take mean of past 30 days for smoothed metrics, or full average if shorter
    lookback = min(30, len(valid_df))
    recent_df = valid_df.tail(lookback)
    
    avg_retail_spread = recent_df['Retail_Spread'].mean()
    avg_buyback_ratio = recent_df['Buyback_Ratio'].mean()
    
    latest_retail_spread = valid_df['Retail_Spread'].iloc[-1]
    latest_buyback_ratio = valid_df['Buyback_Ratio'].iloc[-1]
    
    return {
        'avg_retail_spread': float(avg_retail_spread),
        'avg_buyback_ratio': float(avg_buyback_ratio),
        'latest_retail_spread': float(latest_retail_spread),
        'latest_buyback_ratio': float(latest_buyback_ratio)
    }

def apply_spread_correction(forecast_df, spreads_dict, use_latest=True):
    """
    Applies the calculated spreads to the forecasted spot prices.
    
    Args:
        forecast_df (pd.DataFrame): Forecasted spot prices with columns ['Forecast', 'Lower_Bound', 'Upper_Bound']
        spreads_dict (dict): Dictionary from calculate_historical_spreads()
        use_latest (bool): If True, uses the latest daily spread instead of 30-day average
        
    Returns:
        pd.DataFrame: A copy of forecast_df with added columns:
            - 'Retail_Forecast'
            - 'Retail_Lower_Bound'
            - 'Retail_Upper_Bound'
            - 'Buyback_Forecast'
    """
    corrected_df = forecast_df.copy()
    
    # Pick spread values
    retail_spread = spreads_dict['latest_retail_spread'] if use_latest else spreads_dict['avg_retail_spread']
    buyback_ratio = spreads_dict['latest_buyback_ratio'] if use_latest else spreads_dict['avg_buyback_ratio']
    
    # Apply retail buy price corrections (Spot * (1 + spread))
    corrected_df['Retail_Forecast'] = corrected_df['Forecast'] * (1 + retail_spread)
    corrected_df['Retail_Lower_Bound'] = corrected_df['Lower_Bound'] * (1 + retail_spread)
    corrected_df['Retail_Upper_Bound'] = corrected_df['Upper_Bound'] * (1 + retail_spread)
    
    # Apply buyback corrections (Retail * buyback_ratio)
    corrected_df['Buyback_Forecast'] = corrected_df['Retail_Forecast'] * buyback_ratio
    
    # Round all retail forecasts to nearest 1,000 IDR to mimic real currency increments
    for col in ['Retail_Forecast', 'Retail_Lower_Bound', 'Retail_Upper_Bound', 'Buyback_Forecast']:
        corrected_df[col] = np.round(corrected_df[col], -3)
        
    return corrected_df

if __name__ == "__main__":
    # Test script output
    from data_pipeline import load_and_sync_data
    from forecaster import GoldForecaster
    
    df = load_and_sync_data()
    spreads = calculate_historical_spreads(df)
    print("--- CALCULATED SPREADS ---")
    for k, v in spreads.items():
        print(f"{k}: {v:.6f}")
        
    forecaster = GoldForecaster()
    forecaster.fit(df.set_index('Date')['Spot_IDR_Gram'])
    fc_df = forecaster.forecast(steps=7)
    
    corrected_fc = apply_spread_correction(fc_df, spreads)
    print("\n--- 7 DAY CORRECTED RETAIL FORECAST ---")
    print(corrected_fc[['Retail_Forecast', 'Retail_Lower_Bound', 'Retail_Upper_Bound', 'Buyback_Forecast']])
