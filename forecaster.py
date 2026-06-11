import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress convergence warnings to keep logs clean
warnings.filterwarnings("ignore")

class GoldForecaster:
    """
    SARIMAX forecaster for Gold prices.
    Takes a historical Pandas Series of spot prices and projects future values.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        """
        Initialize forecaster with SARIMAX order.
        Default order is (1, 1, 1) to model non-stationary financial asset walks.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res = None
        self.history = None

    def fit(self, series):
        """
        Fits the SARIMAX model to the spot price history.
        Handles missing days and enforces daily frequency via forward filling.
        """
        # Align series index to datetime
        self.history = series.copy()
        self.history.index = pd.to_datetime(self.history.index)
        
        # Ensure daily frequency
        self.history = self.history.asfreq('D').ffill().bfill()
        
        try:
            # Fit SARIMAX
            model = SARIMAX(
                self.history,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_res = model.fit(disp=False)
            return self
        except Exception as e:
            print(f"SARIMAX fitting error: {e}. Attempting simplified fallback model...")
            # Fallback to simple random walk with drift or ARIMA(1, 1, 0)
            try:
                model = SARIMAX(
                    self.history,
                    order=(1, 1, 0),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.model_res = model.fit(disp=False)
                return self
            except Exception as e_fallback:
                print(f"Fallback ARIMA also failed: {e_fallback}. Using drift projection.")
                self.model_res = None
                return self

    def forecast(self, steps=7):
        """
        Generates out-of-sample forecasts for D+1 to D+7.
        Returns:
            pd.DataFrame: Columns 'Forecast', 'Lower_Bound', 'Upper_Bound' with Date index
        """
        if self.history is None:
            raise ValueError("Forecaster must be fitted before running a forecast.")

        last_date = self.history.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')

        # If model successfully fit
        if self.model_res is not None:
            try:
                forecast_res = self.model_res.get_forecast(steps=steps)
                forecast_mean = forecast_res.predicted_mean
                conf_int = forecast_res.conf_int()
                
                # Make sure lengths match and assign future dates
                forecast_df = pd.DataFrame(index=future_dates)
                forecast_df['Forecast'] = forecast_mean.values
                forecast_df['Lower_Bound'] = conf_int.iloc[:, 0].values
                forecast_df['Upper_Bound'] = conf_int.iloc[:, 1].values
                
                # Verify bounds and sanity check
                # Gold prices should not go below zero
                forecast_df = forecast_df.clip(lower=0)
                return forecast_df
            except Exception as e:
                print(f"Error generating model forecast: {e}. Falling back to drift method.")
                # Fall through to drift projection

        # Drift method fallback (last value + average daily difference)
        last_val = self.history.iloc[-1]
        diffs = self.history.diff().dropna()
        avg_drift = diffs.mean() if len(diffs) > 0 else 0
        std_diff = diffs.std() if len(diffs) > 1 else last_val * 0.01

        forecast_vals = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(1, steps + 1):
            pred = last_val + i * avg_drift
            # Standard error grows with square root of time
            se = np.sqrt(i) * std_diff
            forecast_vals.append(pred)
            lower_bounds.append(pred - 1.96 * se)
            upper_bounds.append(pred + 1.96 * se)
            
        forecast_df = pd.DataFrame(index=future_dates)
        forecast_df['Forecast'] = forecast_vals
        forecast_df['Lower_Bound'] = lower_bounds
        forecast_df['Upper_Bound'] = upper_bounds
        forecast_df = forecast_df.clip(lower=0)
        
        return forecast_df

if __name__ == "__main__":
    # Test script output
    from data_pipeline import load_and_sync_data
    df = load_and_sync_data()
    
    forecaster = GoldForecaster()
    forecaster.fit(df.set_index('Date')['Spot_IDR_Gram'])
    fc_df = forecaster.forecast(steps=7)
    print("--- 7 DAY SPOT FORECAST ---")
    print(fc_df)
