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

    def reset_brain(self, dropout, learning_rate):
        self.setup_brain(self.filter_length, dropout, learning_rate, self.batch_size, self.epochs, frozen = False)

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
    
    def learn(self, shuffle = True):
        train_loader, val_loader = self.prepare_loader(shuffle)
        loss_val, dist_val = self.train(train_loader, val_loader)
        return loss_val, dist_val
            
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
        relative_variations = relative_variations.type(torch.long).to(device = self.device).squeeze()
        prediction = self.net(states, scales)
        prediction = torch.nn.Softmax(1)(prediction)[:,0]

        prediction = self.CDF_m1(prediction)

        dist = 1
        if check_pred:
            dist = torch.mean(torch.abs(relative_variations - prediction))

        return prediction, dist

    def prepare_loader(self, shuffle = True):
        pairs, states, scales, relative_variations = map(torch.stack, zip(*self.memory))

        if shuffle:
            n = states.size(0)
            shuffle_ind = torch.randperm(n)

            states = states[shuffle_ind]
            scales = scales[shuffle_ind]
            relative_variations = relative_variations[shuffle_ind]

        cut_val = int(int(len(self.memory)) * 5/6)

        #here I fit on the last 2/3 and check on the first 1/3
        train_ds = torch.utils.data.TensorDataset(states[:cut_val], scales[:cut_val], relative_variations[:cut_val])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        val_ds = torch.utils.data.TensorDataset(states[cut_val:], scales[cut_val:], relative_variations[cut_val:])
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)

        return train_loader, val_loader
    
    def train_one_epoch(self, train_loader):
        total_loss = 0
        total_dist = []

        for i, data in enumerate(train_loader):
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

        return total_loss/len(train_loader), total_dist

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

    def train(self, train_loader, val_loader):
        
        for epoch in range(self.epochs):
            self.net.train(True)
            loss_train, dist_train = self.train_one_epoch(train_loader)
            print(f"epoch {epoch}: train loss, dist: {np.round(loss_train,4)}, {np.round(dist_train*100,3)}")
            self.loss_train.append(loss_train)
            self.dist_train.append(dist_train)
    
            self.net.train(False)
            loss_val, dist_val = self.check_val(val_loader)
            print(f"epoch {epoch}: val   loss, dist: {np.round(loss_val,4)}, {np.round(dist_val*100,3)}")
            print("____")
            self.loss_val.append(loss_val)
            self.dist_val.append(dist_val)

        return loss_val, dist_val

    def write(self, day, dist):
        with open(self.wallet_log, 'a') as file:
            file.write(f"{day}\t{self.wallet}\t{dist}\n")