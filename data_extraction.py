import pandas as pd
import numpy as np
import yfinance as yf
import time
from matplotlib import pyplot as plt

class Candles(object):
    def __init__(self, data, ticker):
        self.ticker = ticker
        self.data = data
        self._keep_prices_volume_columns()
        self.add_tickername_on_columns()

    def _keep_prices_volume_columns(self):
        self.data = self.data.loc[:,['Open', 'High','Low','Close','Volume']]

    def get_percentage_of_NaN(self):
        number_of_NaN = self.data.isna().sum().sum()
        number_of_elements = self.data.size
        return number_of_NaN/number_of_elements*100
    
    def _get_max_consecutive_nan_column(self, column):
        max_consecutive_nan = 0
        current_consecutive_nan = 0

        for value in column:
            if pd.isna(value):
                current_consecutive_nan += 1
                max_consecutive_nan = max(max_consecutive_nan, current_consecutive_nan)
            else:
                current_consecutive_nan = 0

        return max_consecutive_nan

    def get_max_consecutive_nan(self):
        max_consecutive_nan = self.data.apply(self._get_max_consecutive_nan_column, axis=0).max()
        return max_consecutive_nan
    
    def get_max_mean_date_gap(self):
        delta_time = self.data.index.to_series().diff()
        max_gap = delta_time.max()
        mean_gap = delta_time.mean()

        max_gap_hours = np.round(max_gap.total_seconds()/3600/24,2)
        mean_gap_hours = np.round(mean_gap.total_seconds()/3600/24,2)

        return max_gap_hours, mean_gap_hours

    def check_data_quality(self):
        if self.data.size < 10:
            print("Empty dataset")
        else:
            percentage_of_nan = self.get_percentage_of_NaN()
            consecutive_nan = self.get_max_consecutive_nan()
            max_date_gap, mean_date_gap = self.get_max_mean_date_gap()

            print(f"Percentage of NaN: {np.round(percentage_of_nan,2)} %")
            print(f"Consecutive NaN: {consecutive_nan}")
            print(f"Maximum gap between two dates: {max_date_gap} days")
            print(f"Average gap between two dates: {mean_date_gap} days")

    def clean_data(self):
        self.data = self.data.fillna(method='ffill')

    def add_tickername_on_columns(self):
        header = self.data.columns
        new_header = [(self.ticker, h) for h in header]
        self.data.columns = pd.MultiIndex.from_tuples(new_header)
    
    def plot(self, color):
        data = self.data[self.ticker]
        plt.plot(data.Open, color = color, lw = 2)
        

class Sectors(object):
    def __init__(self, sector = "REIT - Residential"):
        self.available_sectors = self.get_available_sectors()
        self.sector = sector

    def set_sector(self, sector):
        if sector in self.available_sectors:
            self.sector = sector
        else:
            print(f"{sector} is not an available sector")

    def get_available_sectors(self):
        all_symbols_file = "data\\YahooTickerSymbols_2017_09.csv"
        sectors = []
        
        with open(all_symbols_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = line.split(";")
                ticker, company_name, exchange, sector, country = data

                is_in_usa = ( country == "USA" )

                if is_in_usa and sector not in sectors:
                    sectors.append(sector)
        
        return sectors

    def _get_tickers_in_sector(self):
        all_symbols_file = "data\\YahooTickerSymbols_2017_09.csv"
        tickers_list = []
        
        with open(all_symbols_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = line.split(";")
                ticker, company_name, exchange, sector, country = data

                is_in_usa = ( country == "USA" )
                is_target_sector = ( sector == self.sector )

                if is_in_usa and is_target_sector:
                    tickers_list.append(ticker)
        
        return tickers_list

    def _get_volumes(self, tickers_list):
        volume_list = []
        for ticker in tickers_list:
            time.sleep(0.1)
            candle = yf.Ticker(ticker).history(interval = "1d", period = "6mo")
            mean_day_price = (candle.Open + candle.Close + candle.High + candle.Low) / 4
            volume_dollar = candle.Volume * mean_day_price
            mean_volume_dollar = volume_dollar.mean()
            volume_list.append(mean_volume_dollar)
        return volume_list

    def sort_by_volume(self):     
        tickers_list = self._get_tickers_in_sector()
        volume_list = self._get_volumes(tickers_list)
            
        df = pd.DataFrame({"tickers" : tickers_list, "volume" : volume_list})
        df = df.sort_values(by='volume', ascending=False)
        df = df.dropna()

        save_path = f"data/{self.sector}_volume.csv"
        df.to_csv(save_path, index = False)

        print("---")
        print(f"Sorted stocks by volume saved in {save_path}")
        print(f"Number of tickers found: {df.shape[0]}")
        print("---")
        
        return df

class Ticker(object):
    def __init__(self, ticker):
        self.ticker = ticker
        (self.candles_1h, self.candles_1d, 
         self.candles_1wk) = self.get_candles()
        
    def get_candles(self):
        yf_ticker = yf.Ticker(self.ticker)

        candles_1h = yf_ticker.history(period = "730d", interval = "1h")
        candles_1d = yf_ticker.history(period = "max", interval = "1d")
        candles_1wk = yf_ticker.history(period = "max", interval = "1wk")
        
        candles_1h = Candles(candles_1h, self.ticker)
        candles_1d = Candles(candles_1d, self.ticker)
        candles_1wk = Candles(candles_1wk, self.ticker)

        return candles_1h, candles_1d, candles_1wk

    def check_data_quality(self):
        print("1 hour candles:")
        self.candles_1h.check_data_quality()
        print("\n1 day candles:")
        self.candles_1d.check_data_quality()
        print("\n1 week candles:")
        self.candles_1wk.check_data_quality()
        print("")

    def plot(self):
        self.candles_1h.plot("r")
        self.candles_1d.plot("g")
        self.candles_1wk.plot("b")
        plt.xlabel("date")
        plt.ylabel("Open Price")
        plt.title(self.ticker)
        plt.show()

    def clean(self):
        self.candles_1h.clean_data()
        self.candles_1d.clean_data()
        self.candles_1wk.clean_data()

def add_to_dataframe(df_dict, ticker):
    df_dict["1h"] = pd.concat([df_dict["1h"], ticker.candles_1h.data], axis = 1)
    df_dict["1d"] = pd.concat([df_dict["1d"], ticker.candles_1d.data], axis = 1)
    df_dict["1wk"] = pd.concat([df_dict["1wk"], ticker.candles_1wk.data], axis = 1)
    return df_dict

def ask_user():
    user_response = input("Keep that ticker? (y/n): ")
    if user_response.lower() == 'y':
        print("Ticker kept")
        return True
    print("Ticker discarded\n")
    return False

def swap_levels(df):
    df = df.swaplevel(0,1, axis = 1)
    df.sort_index(axis=1, level = [0,1], inplace = True)
    return df


#0 chose a sector
sector_name = "REIT - Residential"
sector = Sectors(sector_name)

#1 sort by volume
tickers_volume_df = sector.sort_by_volume()
tickers = tickers_volume_df["tickers"]

#2 build dataframe
max_number_tickers = 10
df_dict = { "1h" : pd.DataFrame(), "1d" : pd.DataFrame(), "1wk" : pd.DataFrame() }

added_tickers = 0
for ticker in tickers:
    t = Ticker(ticker)
    t.check_data_quality()
    t.plot()

    keep = ask_user()

    if keep:
        t.clean()
        df_dict = add_to_dataframe(df_dict, t)
        added_tickers+=1
    if added_tickers == 10:
        break

for key in df_dict:
    to_be_saved = df_dict[key].dropna()
    to_be_saved = swap_levels(to_be_saved)
    to_be_saved.reset_index().to_csv(f"data/{sector_name}_{key}.csv", index = False)