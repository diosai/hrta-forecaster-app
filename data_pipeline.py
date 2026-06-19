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
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    @classmethod
    def scrape_live_price(cls):
        """
        Scrapes live retail gold prices.
        """
        try:
            response = requests.get(cls.URL, headers=cls.HEADERS, timeout=15)
            response.raise_for_status()
            html = response.text
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP Error terdeteksi di server: {http_err}")
            return cls._fallback_response()
        except Exception as e:
            print(f"Error fetching live HRTA price: {e}")
            return cls._fallback_response()

        soup = BeautifulSoup(html, "html.parser")
        
        buy_price = None
        sell_price = None
        
        price_meta = soup.find("meta", {"name": "price.amount"})
        if price_meta:
            buy_price = cls._parse_numeric(price_meta.get("content", ""))

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
            buy_match = re.search(r'(?:Beli:?\s*Rp\s*)([\d\.]+)', desc, re.IGNORECASE)
            if buy_match and not buy_price:
                buy_price = cls._parse_numeric(buy_match.group(1))
            
            sell_match = re.search(r'(?:Jual:?\s*Rp\s*)([\d\.]+)', desc, re.IGNORECASE)
            if sell_match:
                sell_price = cls._parse_numeric(sell_match.group(1))

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "date": today_str,
            "success": buy_price is not None and sell_price is not None
        }

    @staticmethod
    def _parse_numeric(text):
        digits = re.sub(r'[^\d]', '', text)
        return int(digits) if digits else None

    @classmethod
    def _fallback_response(cls):
        return {
            "buy_price": None,
            "sell_price": None,
            "date": datetime.date.today().strftime("%Y-%m-%d"),
            "success": False
        }


def fetch_spot_data(period="2y"):
    """
    Downloads historical spot gold price (GC=F) and USD/IDR exchange rate (IDR=X) from yfinance.
    """
    print(f"Fetching historical spot data (tickers: GC=F, IDR=X) for period: {period}...")
    
    gold = yf.Ticker("GC=F")
    fx = yf.Ticker("IDR=X")
    
    gold_df = gold.history(period=period)
    fx_df = fx.history(period=period)
    
    if gold_df.empty or fx_df.empty:
        raise ValueError("Failed to retrieve spot data from yfinance. Please check network connection.")
    
    gold_close = gold_df[['Close']].rename(columns={'Close': 'Gold_USD_Oz'})
    fx_close = fx_df[['Close']].rename(columns={'Close': 'USD_IDR'})
    
    df = pd.merge(gold_close, fx_close, left_index=True, right_index=True, how='outer')
    df = df.sort_index()
    df = df.ffill().bfill()
    
    TROY_OZ_TO_GRAM = 31.1034768
    df['Spot_USD_Gram'] = df['Gold_USD_Oz'] / TROY_OZ_TO_GRAM
    df['Spot_IDR_Gram'] = df['Spot_USD_Gram'] * df['USD_IDR']
    
    return df


def load_and_sync_data(csv_path="data/gold_price_history.csv", period="2y"):
    """
    Loads historical data, synchronizes it with yfinance spot data,
    scrapes today's live HRTA Gold price, and saves/appends to a local CSV file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    spot_df = fetch_spot_data(period=period)
    live_info = HRTAGoldScraper.scrape_live_price()
    
    spot_df.index = spot_df.index.date
    spot_df = spot_df[~spot_df.index.duplicated(keep='last')]
    
    if os.path.exists(csv_path):
        print(f"Loading existing database from {csv_path}...")
        db_df = pd.read_csv(csv_path)
        db_df['Date'] = pd.to_datetime(db_df['Date']).dt.date
        db_df.set_index('Date', inplace=True)
        
        # Sinkronisasi data lama & pengisian baris baru secara aman tanpa memakai dictionary assignment
        for idx in spot_df.index:
            if idx in db_df.index:
                db_df.loc[idx, 'Gold_USD_Oz'] = spot_df.loc[idx, 'Gold_USD_Oz']
                db_df.loc[idx, 'USD_IDR'] = spot_df.loc[idx, 'USD_IDR']
                db_df.loc[idx, 'Spot_USD_Gram'] = spot_df.loc[idx, 'Spot_USD_Gram']
                db_df.loc[idx, 'Spot_IDR_Gram'] = spot_df.loc[idx, 'Spot_IDR_Gram']
            else:
                spot_idr = spot_df.loc[idx, 'Spot_IDR_Gram']
                sim_retail = spot_idr * 1.0499
                sim_buyback = sim_retail * 0.952
                
                db_df.loc[idx, 'Gold_USD_Oz'] = spot_df.loc[idx, 'Gold_USD_Oz']
                db_df.loc[idx, 'USD_IDR'] = spot_df.loc[idx, 'USD_IDR']
                db_df.loc[idx, 'Spot_USD_Gram'] = spot_df.loc[idx, 'Spot_USD_Gram']
                db_df.loc[idx, 'Spot_IDR_Gram'] = spot_idr
                db_df.loc[idx, 'Retail_Price'] = sim_retail
                db_df.loc[idx, 'Buyback_Price'] = sim_buyback
    else:
        print(f"Initializing new database at {csv_path} with simulated historical retail prices...")
        db_df = pd.DataFrame(index=spot_df.index)
        db_df['Gold_USD_Oz'] = spot_df['Gold_USD_Oz']
        db_df['USD_IDR'] = spot_df['USD_IDR']
        db_df['Spot_USD_Gram'] = spot_df['Spot_USD_Gram']
        db_df['Spot_IDR_Gram'] = spot_df['Spot_IDR_Gram']
        
        np.random.seed(42)
        spread_retail = 1.0499
        spread_buyback = 0.952
        
        noise_retail = np.random.normal(0, 3000, len(db_df))
        db_df['Retail_Price'] = (db_df['Spot_IDR_Gram'] * spread_retail) + noise_retail
        db_df['Buyback_Price'] = db_df['Retail_Price'] * spread_buyback
        
        db_df['Retail_Price'] = np.round(db_df['Retail_Price'], -3)
        db_df['Buyback_Price'] = np.round(db_df['Buyback_Price'], -3)

    today_date = datetime.date.today()
    
    if live_info["success"]:
        print(f"Successfully scraped live retail prices for {live_info['date']}: Buy={live_info['buy_price']}, Sell/Buyback={live_info['sell_price']}")
        db_df.loc[today_date, 'Retail_Price'] = live_info['buy_price']
        db_df.loc[today_date, 'Buyback_Price'] = live_info['sell_price']
    else:
        print(f"Scraper gagal mendapatkan data HRTA untuk hari ini ({today_date}). Menggunakan mekanisme fallback margin statis (+4.99%).")
        if today_date not in db_df.index:
            db_df.loc[today_date] = np.nan
            
        if today_date in spot_df.index:
            spot_idr = spot_df.loc[today_date, 'Spot_IDR_Gram']
        else:
            last_idx = spot_df.index[-1]
            spot_idr = spot_df.loc[last_idx, 'Spot_IDR_Gram']
            
        sim_retail = spot_idr * 1.0499
        sim_buyback = sim_retail * 0.952
        
        if pd.isna(db_df.loc[today_date, 'Retail_Price']):
            db_df.loc[today_date, 'Retail_Price'] = sim_retail
            db_df.loc[today_date, 'Buyback_Price'] = sim_buyback

    if today_date in spot_df.index:
        db_df.loc[today_date, 'Gold_USD_Oz'] = spot_df.loc[today_date, 'Gold_USD_Oz']
        db_df.loc[today_date, 'USD_IDR'] = spot_df.loc[today_date, 'USD_IDR']
        db_df.loc[today_date, 'Spot_USD_Gram'] = spot_df.loc[today_date, 'Spot_USD_Gram']
        db_df.loc[today_date, 'Spot_IDR_Gram'] = spot_df.loc[today_date, 'Spot_IDR_Gram']
    else:
        last_idx = spot_df.index[-1]
        db_df.loc[today_date, 'Gold_USD_Oz'] = spot_df.loc[last_idx, 'Gold_USD_Oz']
        db_df.loc[today_date, 'USD_IDR'] = spot_df.loc[last_idx, 'USD_IDR']
        db_df.loc[today_date, 'Spot_USD_Gram'] = spot_df.loc[last_idx, 'Spot_USD_Gram']
        db_df.loc[today_date, 'Spot_IDR_Gram'] = spot_df.loc[last_idx, 'Spot_IDR_Gram']

    db_df = db_df.sort_index()
    db_df = db_df.ffill().bfill()
    
    # 2. Perbaikan pada reset_index tanpa argumen 'names' untuk menjamin kompatibilitas lintas versi Pandas
    db_df.reset_index(inplace=True)
    db_df.to_csv(csv_path, index=False)
    print(f"Database successfully synchronized and saved to {csv_path}.")
    
    return db_df

if __name__ == "__main__":
    res = load_and_sync_data(csv_path="data/gold_price_history.csv")
    print(res.tail())