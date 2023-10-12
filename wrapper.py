import pandas as pd
from numpy import random
import numpy as np
from pathlib import Path
import torch
import yfinance as yf
import itertools
from datetime import timedelta

class Wrapper(object):
    def __init__(self, number_of_observed_candles, sector = "REIT - Residential"):
        self.sector = sector
        # self.number_of_observed_candles = int(np.ceil(number_of_observed_candles/2**3))*2**3
        self.number_of_observed_candles = number_of_observed_candles
        self.tickers_list = self.get_ticker_list()
        self.pairs = self.enumerate_pairs()
        
        print(f"Number of observed candles: {self.number_of_observed_candles}")
        print(f"{len(self.tickers_list)} tickers traded: {' '.join(self.tickers_list)}")

        self.candles_1h = self.load_candles("1h")
        self.candles_1d = self.load_candles("1d")
        self.candles_1wk = self.load_candles("1wk")
    
    def get_number_of_tickers(self):
        return len(self.tickers_list)

    def get_ticker_list(self):
        data_file_1h = f"data/{self.sector}_1h.csv"
        df = pd.read_csv(data_file_1h, header=[0,1], index_col=0, parse_dates=True, infer_datetime_format=True)
        ticker_list =[]
        for column in df.columns:
            ticker = column[1]
            if ticker not in ticker_list:
                ticker_list.append(ticker)
        return ticker_list

    def load_candles(self, interval):
        data_file = f"data/{self.sector}_{interval}.csv"
        candles = pd.read_csv(data_file, header=[0,1], index_col=0, parse_dates=True, infer_datetime_format=True)
        candles.index = pd.to_datetime(candles.index, utc = True).tz_convert('America/New_York')
        candles = candles.loc[:, (['Close', 'Open','High','Low','Volume'], self.tickers_list)]
        return candles

    def get_states(self, day):

        states_1h, scales_1h, day1h_0, day1h_1 = self._get_states_interval(day, "1h")
        states_1d, scales_1d, day1d_0, day1d_1 = self._get_states_interval(day, "1d")
        states_1wk, scales_1wk, day1wk_0, day1wk_1 = self._get_states_interval(day, "1wk")

        states = []
        scales = []
        for i in range(len(self.pairs)):
            state_1h = states_1h[i]
            state_1d = states_1d[i]
            state_1wk = states_1wk[i]
            state = torch.cat((state_1h,state_1d,state_1wk),dim=0)
            states.append(state)
            scale = torch.tensor([scales_1h[i],scales_1d[i],scales_1wk[i]])
            scales.append(scale)

        return states, scales, (day1h_0, day1h_1), (day1d_0, day1d_1), (day1wk_0, day1wk_1)

    def _get_states_interval(self, day, interval):
        states = []
        scales = []

        if interval == "1h":
            raw_data = self.get_n_candles_1h(day)
        if interval == "1d":
            raw_data = self.get_n_candles_1d(day)
        if interval == "1wk":
            raw_data = self.get_n_candles_1wk(day)

        day0 = raw_data.index[0]
        day1 = raw_data.index[-1]

        variation, minmax_price, minmax_vol = self.get_scaling_info(raw_data)
        data = self.convert_to_indice(raw_data, minmax_price, minmax_vol)
        
        for pair in self.pairs:
            state, scale = self.get_pair_state(data, pair, variation)
            states.append(state)
            scales.append(scale)
        
        return states, scales, day0, day1

    def get_n_candles_1h(self, day):
        timestamp = pd.Timestamp(day.strftime('%Y-%m-%d') + ' 10:30:00', tz='America/New_York')
        candles = self.candles_1h[self.candles_1h.index < timestamp ]
        candles = candles.iloc[-self.number_of_observed_candles:]
        return candles

    def get_n_candles_1d(self, day):
        candles = self.candles_1d[self.candles_1d.index.date < day ]
        candles = candles.iloc[-self.number_of_observed_candles:]
        return candles

    def get_n_candles_1wk(self, day):
        candles = self.candles_1wk[self.candles_1wk.index.date <= day - timedelta(days = 7)]
        candles = candles.iloc[-self.number_of_observed_candles:]
        return candles

    def get_scaling_info(self, candles):
        minmax_price =  ( candles.Low.min(), candles.High.max() )
        variation = ( candles.High.max() - candles.Low.min() ) / candles.Low.min()
        minmax_vol = ( candles.Volume.min(), candles.Volume.max() )
        return variation, minmax_price, minmax_vol

    def convert_to_indice(self, candles, minmax_price, minmax_vol):
        O = self._convert_to_indice(candles.Open, minmax_price)
        C = self._convert_to_indice(candles.Close, minmax_price)
        H = self._convert_to_indice(candles.High, minmax_price)
        L = self._convert_to_indice(candles.Low, minmax_price)
        V = self._convert_to_indice(candles.Volume, minmax_vol)
        V = V/2.5
        V = V.astype(int)
        data = ( O, C, H, L, V )
        return data

    def _convert_to_indice(self, val, minmax):
        val = val.fillna(method='ffill')
        val = ( val - minmax[0] ) / ( minmax[1] - minmax[0] ) * 100
        return val

    def get_pair_state(self, data, pair, variation):
        stock1 = self.tickers_list[pair[0]]
        stock2 = self.tickers_list[pair[1]]
        pair = [stock1, stock2]
        
        O = data[0][pair]
        C = data[1][pair]
        H = data[2][pair]
        L = data[3][pair]
        V = data[4][pair]

        variation = variation.loc[pair]
        max_variation = variation.max()
        factor = variation/max_variation

        O = np.asarray(O*factor).astype(int)
        C = np.asarray(C*factor).astype(int)
        H = np.asarray(H*factor).astype(int)
        L = np.asarray(L*factor).astype(int)
        V = np.asarray(V).astype(int)

        data = ( O, C, H, L, V )

        state = self.convert_to_image(data)
        
        return state, max_variation

    def enumerate_pairs(self):
        permutations = itertools.permutations(range(len(self.tickers_list)),2)
        pairs = torch.tensor(list(permutations), dtype = int)
        return pairs

    def convert_to_image(self, data):
        O, C, H, L, V = data

        channels = 2
        offset_price = 101
        offset_volume = 144

        height = 144 #2**3 * 27 allows up to 3 (2,2) maxPool

        state = torch.zeros((channels, height, 4*self.number_of_observed_candles)).to(torch.uint8)

        for j in range(channels):
            for i in range(self.number_of_observed_candles):
                state[j, offset_price - H[i,j]:offset_price - L[i,j], 4*i+1] = 2
                state[j, offset_price - C[i,j]:offset_price - O[i,j], 4*i:4*i+3] = 3
                state[j, offset_price - O[i,j]:offset_price - C[i,j], 4*i:4*i+3] = 1
                if C[i,j] <= O[i,j]:
                    state[j, offset_volume - V[i,j]:height, 4*i:4*i+3] = 1
                else:
                    state[j, offset_volume - V[i,0]:height, 4*i:4*i+3] = 3

        return state

    def get_best_actions(self, day1, day2):
        candles1 = self.candles_1h[self.candles_1h.index.date == day1]
        candles2 = self.candles_1h[self.candles_1h.index.date == day2]

        day1_value = candles1.Close.iloc[0] #val 1h after open
        day2_value = candles2.Close.iloc[0] #val 1h after open

        day1_timestamp = candles1.index[0]
        day2_timestamp = candles2.index[0]
        periode = (day1_timestamp, day2_timestamp)

        variation = ( day2_value - day1_value ) / day1_value
        variation = variation.values

        best_action = []
        for pair in self.pairs:
            if variation[pair[0]] > variation[pair[1]]:
                best_action.append([0])
            else:
                best_action.append([1])
        
        return torch.tensor(best_action), periode

    def get_state_best_action(self, day1, day2, Troubleshoot = False):
        batch = []
        states, scales, periode_1h, periode_1d, periode_1wk = self.get_states(day1)
        best_action, periode_best_action = self.get_best_actions(day1, day2)

        if Troubleshoot:
            print("---")
            print(f"variation is taken between Close of 1h candle at {periode_best_action[0]},")
            print(f"and Close of 1h candle starting at {periode_best_action[1]} \n")
            print(f"1h candle plate between:\n1: {periode_1h[0]}\n2: {periode_1h[1]} \n")
            print(f"1d candle plate between:\n1: {periode_1d[0]}\n2: {periode_1d[1]} \n")
            print(f"1wk candle plate between:\n1: {periode_1wk[0]}\n2: {periode_1wk[1]} \n")
            print("(add 1h, 1d and 7d to last candles respectively to get close time of last candle)")
            print("---")
        
        for p, st, sc, ba in zip(self.pairs, states, scales, best_action):
            batch.append((p, st, sc, ba))
    
        return batch

    def get_available_days(self):
        days = sorted(list(set(self.candles_1h.index.date)))
        return days

    def market_price(self, day):
        day_val = self.candles_1h[self.candles_1h.index.date == day]
        return day_val.Close.iloc[0]

    def order_passed(self, day1, day2, ticker, strike, is_long):
        timestamp1 = pd.Timestamp(day1.strftime('%Y-%m-%d') + ' 9:30:00', tz='America/New_York')
        timestamp2 = pd.Timestamp(day2.strftime('%Y-%m-%d') + ' 9:30:00', tz='America/New_York')
        
        candles = self.candles_1h[[("High",ticker),("Low",ticker)]]
        candles = candles[candles.index >= timestamp1 ]
        candles = candles[candles.index < timestamp2 ]

        minprice = candles["Low"].min().min()
        maxprice = candles["High"].max().max()

        if is_long and strike > minprice :
            return True
        if not is_long and strike < maxprice:
            return True

        return False
    
    # def get_training_data(self):
    #     self.load_history()

    #     days = sorted(list(set(self.candles.index.date)))[21:]

    #     memory=[]
    #     for day in days:
    #         state, prediction = self.get_training_data_one_day(day)
    #         state[torch.isnan(state)] = 1e-6
    #         memory.append([state, prediction])
    
    #     return memory