from wrapper_variation import Wrapper
from neuralnet import AlexNetLike
import torch
import numpy as np
from collections import deque
import itertools
import os
import datetime

def loss_custom(prediction, target):
    epsilon = 1e-9 #to avoid issues in 0
    pred = torch.nn.Softmax(1)(prediction)[:,0]
    dist = torch.abs(pred-target)
    loss = -torch.log(1-dist+epsilon)
    return torch.mean(loss)

class Trader(object):
    def __init__(self, slope,  holding_time, number_of_observed_candles, days_in_memory, filter_length, dropout, learning_rate, batch_size, epochs, frozen = False, metadata = None):
        
        self.holding_time = holding_time #days
        self.metadata = metadata
        self.slope = slope

        self.setup_market(number_of_observed_candles)
        self.setup_memory(days_in_memory)
        self.setup_brain(filter_length, dropout, learning_rate, batch_size, epochs, frozen)
        self.setup_wallet()
        
    def CDF(self, x):
        return 0.5*(1+torch.tanh(self.slope*x))
    
    def CDF_m1(self, x):
        return 1/self.slope * torch.arctanh(2*x-1)

    def setup_market(self, number_of_observed_candles):
        self.market = Wrapper(number_of_observed_candles)
        self.number_of_observed_candles = self.market.number_of_observed_candles
        self.number_of_tickers = self.market.get_number_of_tickers()

    def setup_memory(self, days_in_memory):
        self.days_in_memory = days_in_memory
        self.memory_size = len(self.market.pairs) * days_in_memory
        self.memory = deque(maxlen = self.memory_size)

    def setup_brain(self, filter_length, dropout, learning_rate, batch_size, epochs, frozen):
        self.epochs = epochs
        self.batch_size = batch_size
        self.state_dim = (6, 144, self.number_of_observed_candles*4)
        self.action_dim = 2
        self.dropout = dropout

        self.filter_length = filter_length
        self.device = "cuda"
        self.net = AlexNetLike(self.state_dim, self.action_dim, filter_length, dropout, frozen).float()
        self.net = self.net.to(device=self.device)
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay = 1e-5)
        self.loss_fn = loss_custom

        self.loss_train = []
        self.loss_val = []
        self.dist_train = []
        self.dist_val = []
    
    def setup_wallet(self):
        self.wallet_log = f"backtests_variation\\Log\\run_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}.csv"
        if not os.path.exists(self.wallet_log):
            with open(self.wallet_log, 'a') as file:
                (days_in_memory, lr, dropout, brain, threshold) = self.metadata
                file.write(f"#days in memory: {days_in_memory}\n")
                file.write(f"#lr: {lr}\n")
                file.write(f"#dropout: {dropout}\n")
                file.write(f"#brain: {brain}\n")
                file.write(f"#threshold: {threshold}\n")
                file.write(f"#slope: {self.slope}\n")
                file.write("#day\twallet\tdist\n")
        self.wallet = 1
        self.holding = {}
        for i in range(self.number_of_tickers):
            self.holding[i] = 0

    def reset_brain(self, dropout, learning_rate, slope, filter_length):
        self.filter_length = filter_length
        self.setup_brain(self.filter_length, dropout, learning_rate, self.batch_size, self.epochs, frozen = False)
        self.slope = slope

    def observe_past(self, day_open_position, day_close_position, Troubleshoot = False):
        batch = self.market.get_state_relative_variation(day_open_position, day_close_position, Troubleshoot)
        return batch

    def observe_present(self, day):
        plates, scales = self.market.get_states(day)
        pairs = self.market.pairs

        observation = []
        for pair, plate, scale in zip(pairs, plates, scales):
            observation.append((pair, plate, scale))
    
        return observation

    def memorize(self, batch):
        self.memory.extend(batch)
            
    @torch.no_grad()
    def predict(self, state, scale):
        self.net.train(False)
        prediction = self.net(state, scale)
        prediction = torch.nn.Softmax(1)(prediction)[0]
        return prediction

    @torch.no_grad()
    def predict_batch(self, batch, check_pred = False):
        self.net.train(False)
        pairs, states, scales, relative_variations = map(torch.stack, zip(*batch))

        states = states.float()
        states = states.to(device = self.device)
        scales = scales.float()
        scales = scales.to(device = self.device)
        relative_variations = relative_variations.to(device = self.device)

        prediction = self.net(states, scales)

        prediction = torch.nn.Softmax(1)(prediction)[:,0]

        prediction = self.CDF_m1(prediction)

        dist = 1
        if check_pred:
            dist = torch.mean(torch.abs(relative_variations - prediction))
            max_gain = relative_variations.max() - relative_variations.min()
            print(f"maximum gain = {float(max_gain)*100} %")

        return prediction, float(dist)

    def prepare_loader_noval(self):
        pairs, states, scales, relative_variations = map(torch.stack, zip(*self.memory))

        train_ds = torch.utils.data.TensorDataset(states, scales, relative_variations)
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)
        

    def prepare_loader(self, shuffle = True):
        pairs, states, scales, relative_variations = map(torch.stack, zip(*self.memory))

        if shuffle:
            n = states.size(0)
            shuffle_ind = torch.randperm(n)

            states = states[shuffle_ind]
            scales = scales[shuffle_ind]
            relative_variations = relative_variations[shuffle_ind]

        # cut_val = int(int(len(self.memory)) * 3/4)
        cut1 = int(len(self.memory) * 1/8)
        cut2 = int(len(self.memory) * 7/8)


        #here I fit on the last 2/3 and check on the first 1/3
        train_ds = torch.utils.data.TensorDataset(states[cut1:cut2], scales[cut1:cut2], relative_variations[cut1:cut2])
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        states_val = torch.cat((states[:cut1], states[cut2:]), dim = 0)
        scales_val = torch.cat((scales[:cut1], scales[cut2:]), dim = 0)
        relative_variations_val = torch.cat((relative_variations[:cut1], relative_variations[cut2:]), dim = 0)

        val_ds = torch.utils.data.TensorDataset(states_val, scales_val, relative_variations_val)
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # cut_val = 5 * 90
        # train_ds = torch.utils.data.TensorDataset(states[cut_val:], scales[cut_val:], relative_variations[cut_val:])
        # self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        # val_ds = torch.utils.data.TensorDataset(states[:cut_val], scales[:cut_val], relative_variations[:cut_val])
        # self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)
    
    def train_one_epoch(self):
        total_loss = 0
        total_dist = []

        for i, data in enumerate(self.train_loader):
            state, scale, relative_variation = data
            state = state.float()
            state = state.to(device = self.device)
            scale = scale.float()
            scale = scale.to(device = self.device)
            relative_variation = relative_variation.to(device = self.device)
            relative_variation = self.CDF(relative_variation) #project to [0,1]

            self.optimizer.zero_grad()
            prediction = self.net(state, scale)

            loss = self.loss_fn(prediction, relative_variation) #prediction is pushed on [0,1] in the loss like for CrossEntropy

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            prediction = torch.nn.Softmax(1)(prediction)[:,0] #project on [0,1]

            #project back to the relative variations:
            prediction = self.CDF_m1(prediction) 
            true_value = self.CDF_m1(relative_variation)
            dist = torch.abs( prediction - true_value )

            total_dist.append( torch.mean(dist) )

        total_dist = torch.mean(torch.tensor(total_dist))

        return total_loss/len(self.train_loader), total_dist

    @torch.no_grad()
    def check_val(self, val_loader):
        self.net.train(False)
        total_loss = 0
        total_dist = []

        for i, data in enumerate(val_loader):
            state, scale, relative_variation = data
            state = state.float()
            state = state.to(device = self.device)
            scale = scale.float()
            scale = scale.to(device = self.device)
           
            relative_variation = relative_variation.to(device = self.device)
            relative_variation = self.CDF(relative_variation) #project to [0,1]

            prediction = self.net(state, scale)

            loss = self.loss_fn(prediction, relative_variation) 
            
            total_loss += loss.item()

            prediction = torch.nn.Softmax(1)(prediction)[:,0] #project on [0,1]
            
            #project back to the relative variations:
            prediction = self.CDF_m1(prediction) 
            true_value = self.CDF_m1(relative_variation)
            dist = torch.abs( prediction - true_value )

            total_dist.append( torch.mean(dist) )

        total_dist = torch.mean(torch.tensor(total_dist))
        
        return total_loss/len(val_loader) , total_dist

    def learn(self):
        for epoch in range(self.epochs):
            self.net.train(True)
            loss_train, dist_train = self.train_one_epoch()
            print(f"epoch {epoch}: train loss, dist: {np.round(loss_train,4)}, {np.round(dist_train*100,3)}")
            self.loss_train.append(loss_train)
            self.dist_train.append(dist_train)
    
            self.net.train(False)
            loss_val, dist_val = self.check_val(self.val_loader)
            print(f"epoch {epoch}: val   loss, dist: {np.round(loss_val,4)}, {np.round(dist_val*100,3)}")
            print("____")
            self.loss_val.append(loss_val)
            self.dist_val.append(dist_val)

        return loss_val, dist_val
    
    def learn_noval(self):
        for epoch in range(self.epochs):
            self.net.train(True)
            loss_train, dist_train = self.train_one_epoch()
            print(f"epoch {epoch}: train loss, dist: {np.round(loss_train,4)}, {np.round(dist_train*100,3)}")
            self.loss_train.append(loss_train)
            self.dist_train.append(dist_train)

        return loss_train, dist_train

    def act(self, scores, day, threshold):
        price = self.market.market_price(day)

        for key in self.holding:
            self.wallet += self.holding[key] * price[key]
            self.holding[key] = 0

        invest_val = self.wallet/2

        argsort_scores = np.argsort(scores)
        scores_sorted = scores[argsort_scores]
        n_pairs = (self.number_of_tickers - 1)
        max_diff = (scores_sorted[-1] - scores_sorted[0]) / n_pairs
        print(f"gain predicted: {np.round(max_diff*100,2)}%")

        if max_diff > threshold:
            arg_long = argsort_scores[-1]
            arg_short = argsort_scores[0]
            self.holding[arg_short] = - invest_val / price[arg_short]
            self.holding[arg_long] = + invest_val / price[arg_long]

    def write(self, day, dist):
        with open(self.wallet_log, 'a') as file:
            file.write(f"{day}\t{self.wallet}\t{dist}\n")

    def save(self, save_path):
        torch.save(
            dict(model=self.net.state_dict()),
            save_path,
        )
        
        print(f"Net saved to {save_path}")
    
    def load(self, model):
        chkpt = torch.load(model)
        self.net.load_state_dict(chkpt['model'])