import os
import re
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup

class HRTAGoldScraper:
    """
    Scraper to retrieve the live gold retail buy and sell (buyback) prices
    from the official HRTA Gold price page.
    """
    URL = "https://hrtagold.id/id/gold-price"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    @classmethod
    def scrape_live_price(cls):
        """
        Scrapes live retail gold prices.
        Returns:
            dict: {
                "buy_price": int or None,
                "sell_price": int or None,
                "date": str (YYYY-MM-DD),
                "raw_description": str
            }
        """
        try:
            response = requests.get(cls.URL, headers=cls.HEADERS, timeout=15)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            print(f"Error fetching live HRTA price: {e}")
            return cls._fallback_response()

        soup = BeautifulSoup(html, "html.parser")
        
        # Parse from Meta Tags
        buy_price = None
        sell_price = None
        
        # 1. Try price.amount for buy price
        price_meta = soup.find("meta", {"name": "price.amount"})
        if price_meta:
            buy_price = cls._parse_numeric(price_meta.get("content", ""))

        # 2. Try description tags for both prices
        desc_meta = soup.find("meta", {"name": "description"})
        og_desc_meta = soup.find("meta", {"property": "og:description"})
        
        descriptions = []
        if desc_meta:
            descriptions.append(desc_meta.get("content", ""))
        if og_desc_meta:
            descriptions.append(og_desc_meta.get("content", ""))

        for desc in descriptions:
            if not desc:
                continue
            # Regex to find Beli (Buy) price, e.g. "Beli: Rp 2.521.000" or "Beli Rp 2.521.000"
            buy_match = re.search(r'(?:Beli:?\s*Rp\s*)([\d\.]+)', desc, re.IGNORECASE)
            if buy_match and not buy_price:
                buy_price = cls._parse_numeric(buy_match.group(1))
            
            # Regex to find Jual (Sell/Buyback) price, e.g. "Jual Rp 2.400.000" or "Jual: Rp 2.400.000"
            sell_match = re.search(r'(?:Jual:?\s*Rp\s*)([\d\.]+)', desc, re.IGNORECASE)
            if sell_match:
                sell_price = cls._parse_numeric(sell_match.group(1))

        # Date of update
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "date": today_str,
            "success": buy_price is not None and sell_price is not None
        }

    @staticmethod
    def _parse_numeric(text):
        """Helper to extract numeric digits from price strings (e.g. 'Rp 2.521.000' -> 2521000)"""
        digits = re.sub(r'[^\d]', '', text)
        return int(digits) if digits else None

    @classmethod
    def _fallback_response(cls):
        """Returns a structure indicating failure"""
        return {
            "buy_price": None,
            "sell_price": None,
            "date": datetime.date.today().strftime("%Y-%m-%d"),
            "success": False
        }


def fetch_spot_data(period="2y"):
    """
    Downloads historical spot gold price (GC=F) and USD/IDR exchange rate (IDR=X) from yfinance,
    aligns their timelines, and calculates the Spot price in IDR per gram.
    
    Calculation:
        Spot_USD_Gram = Spot_Gold_USD_Ounce / 31.1034768
        Spot_IDR_Gram = Spot_USD_Gram * USD_IDR_Exchange_Rate
    """
    print(f"Fetching historical spot data (tickers: GC=F, IDR=X) for period: {period}...")
    
    # Fetch data
    gold = yf.Ticker("GC=F")
    fx = yf.Ticker("IDR=X")
    
    gold_df = gold.history(period=period)
    fx_df = fx.history(period=period)
    
    if gold_df.empty or fx_df.empty:
        raise ValueError("Failed to retrieve spot data from yfinance. Please check network connection.")
    
    # Keep only Close prices
    gold_close = gold_df[['Close']].rename(columns={'Close': 'Gold_USD_Oz'})
    fx_close = fx_df[['Close']].rename(columns={'Close': 'USD_IDR'})
    
    # Merge on Date index
    df = pd.merge(gold_close, fx_close, left_index=True, right_index=True, how='outer')
    
    # Sort index and forward fill gaps (holidays/weekend mismatch)
    df = df.sort_index()
    df = df.ffill().bfill()
    
    # Constants
    TROY_OZ_TO_GRAM = 31.1034768
    
    # Calculate IDR Spot Price per Gram
    df['Spot_USD_Gram'] = df['Gold_USD_Oz'] / TROY_OZ_TO_GRAM
    df['Spot_IDR_Gram'] = df['Spot_USD_Gram'] * df['USD_IDR']
    
    return df


def load_and_sync_data(csv_path="data/gold_price_history.csv", period="2y"):
    """
    Loads historical data, synchronizes it with yfinance spot data,
    scrapes today's live HRTA Gold price, and saves/appends to a local CSV file.
    
    If the CSV does not exist, it initializes it by simulating historical retail prices
    based on historical spot prices + retail spread (4.99%) + buyback spread (~4.8%).
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 1. Fetch yfinance spot data
    spot_df = fetch_spot_data(period=period)
    
    # 2. Scrape live retail prices
    live_info = HRTAGoldScraper.scrape_live_price()
    
    # Normalize Spot index to date strings or datetime.date
    spot_df.index = spot_df.index.date
    spot_df = spot_df[~spot_df.index.duplicated(keep='last')]
    
    # 3. Handle CSV caching
    if os.path.exists(csv_path):
        print(f"Loading existing database from {csv_path}...")
        db_df = pd.read_csv(csv_path)
        db_df['Date'] = pd.to_datetime(db_df['Date']).dt.date
        db_df.set_index('Date', inplace=True)
        
        # Merge spot data updates
        for idx in spot_df.index:
            if idx in db_df.index:
                db_df.loc[idx, 'Gold_USD_Oz'] = spot_df.loc[idx, 'Gold_USD_Oz']
                db_df.loc[idx, 'USD_IDR'] = spot_df.loc[idx, 'USD_IDR']
                db_df.loc[idx, 'Spot_USD_Gram'] = spot_df.loc[idx, 'Spot_USD_Gram']
                db_df.loc[idx, 'Spot_IDR_Gram'] = spot_df.loc[idx, 'Spot_IDR_Gram']
            else:
                # New spot data row, simulate physical prices
                spot_idr = spot_df.loc[idx, 'Spot_IDR_Gram']
                # Average spread is 4.99% for retail
                sim_retail = spot_idr * 1.0499
                # Buyback is roughly 4.8% below spot or 95.2% of retail
                sim_buyback = sim_retail * 0.952
                
                db_df.loc[idx] = {
                    'Gold_USD_Oz': spot_df.loc[idx, 'Gold_USD_Oz'],
                    'USD_IDR': spot_df.loc[idx, 'USD_IDR'],
                    'Spot_USD_Gram': spot_df.loc[idx, 'Spot_USD_Gram'],
                    'Spot_IDR_Gram': spot_idr,
                    'Retail_Price': sim_retail,
                    'Buyback_Price': sim_buyback
                }
    else:
        print(f"Initializing new database at {csv_path} with simulated historical retail prices...")
        db_df = pd.DataFrame(index=spot_df.index)
        db_df['Gold_USD_Oz'] = spot_df['Gold_USD_Oz']
        db_df['USD_IDR'] = spot_df['USD_IDR']
        db_df['Spot_USD_Gram'] = spot_df['Spot_USD_Gram']
        db_df['Spot_IDR_Gram'] = spot_df['Spot_IDR_Gram']
        
        # Simulate historical retail prices
        np.random.seed(42) # Keep simulations reproducible
        spread_retail = 1.0499
        spread_buyback = 0.952
        
        # Add slight noise to make history look realistic
        noise_retail = np.random.normal(0, 3000, len(db_df))
        db_df['Retail_Price'] = (db_df['Spot_IDR_Gram'] * spread_retail) + noise_retail
        db_df['Buyback_Price'] = db_df['Retail_Price'] * spread_buyback
        
        # Round prices to match typical currency formatting (thousands)
        db_df['Retail_Price'] = np.round(db_df['Retail_Price'], -3)
        db_df['Buyback_Price'] = np.round(db_df['Buyback_Price'], -3)

    # 4. Sync live scraped data if successful
    if live_info["success"]:
        today_date = datetime.datetime.strptime(live_info["date"], "%Y-%m-%d").date()
        print(f"Successfully scraped live retail prices for {today_date}: Buy={live_info['buy_price']}, Sell/Buyback={live_info['sell_price']}")
        
        # Update/insert live price row
        db_df.loc[today_date, 'Retail_Price'] = live_info['buy_price']
        db_df.loc[today_date, 'Buyback_Price'] = live_info['sell_price']
        
        # If today's spot values aren't in the database, grab last available spot values
        if today_date not in db_df.index or pd.isna(db_df.loc[today_date, 'Spot_IDR_Gram']):
            if today_date in spot_df.index:
                db_df.loc[today_date, 'Gold_USD_Oz'] = spot_df.loc[today_date, 'Gold_USD_Oz']
                db_df.loc[today_date, 'USD_IDR'] = spot_df.loc[today_date, 'USD_IDR']
                db_df.loc[today_date, 'Spot_USD_Gram'] = spot_df.loc[today_date, 'Spot_USD_Gram']
                db_df.loc[today_date, 'Spot_IDR_Gram'] = spot_df.loc[today_date, 'Spot_IDR_Gram']
            else:
                # Grab the last known spot values
                last_idx = spot_df.index[-1]
                db_df.loc[today_date, 'Gold_USD_Oz'] = spot_df.loc[last_idx, 'Gold_USD_Oz']
                db_df.loc[today_date, 'USD_IDR'] = spot_df.loc[last_idx, 'USD_IDR']
                db_df.loc[today_date, 'Spot_USD_Gram'] = spot_df.loc[last_idx, 'Spot_USD_Gram']
                db_df.loc[today_date, 'Spot_IDR_Gram'] = spot_df.loc[last_idx, 'Spot_IDR_Gram']

    # Sort data by Date index
    db_df = db_df.sort_index()
    # Forward fill spot components if there are gaps on weekends/holidays where retail has data
    db_df = db_df.ffill().bfill()
    
    # Save back to CSV
    db_df.reset_index(names='Date', inplace=True)
    db_df.to_csv(csv_path, index=False)
    print(f"Database successfully synchronized and saved to {csv_path}.")
    
    return db_df

if __name__ == "__main__":
    # Test script output
    res = load_and_sync_data(csv_path="data/gold_price_history.csv")
    print(res.tail())
