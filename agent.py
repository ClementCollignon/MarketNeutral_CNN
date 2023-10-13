from wrapper import Wrapper
from neuralnet import AlexNetLike

import torch
import numpy as np

from collections import deque
import itertools

import os
import datetime


class Trader(object):
    def __init__(self, holding_time, number_of_observed_candles, days_in_memory, filter_length, dropout, learning_rate, batch_size, epochs, pretrained = False, metadata = None):
        
        self.holding_time = holding_time #days
        self.metadata = metadata

        self.setup_market(number_of_observed_candles)
        self.setup_memory(days_in_memory)
        self.setup_brain(filter_length, dropout, learning_rate, batch_size, epochs, pretrained)
        self.setup_wallet()
        

    def setup_market(self, number_of_observed_candles):
        self.market = Wrapper(number_of_observed_candles)
        self.number_of_observed_candles = self.market.number_of_observed_candles
        self.number_of_tickers = self.market.get_number_of_tickers()

    def setup_memory(self, days_in_memory):
        self.days_in_memory = days_in_memory
        self.memory_size = len(self.market.pairs) * days_in_memory
        self.memory = deque(maxlen = self.memory_size)

    def setup_brain(self, filter_length, dropout, learning_rate, batch_size, epochs, pretrained):
        self.epochs = epochs
        self.batch_size = batch_size
        self.state_dim = (6, 144, self.number_of_observed_candles*4)
        self.action_dim = 2
        self.dropout = dropout

        self.filter_length = filter_length
        self.device = "cuda"
        self.net = AlexNetLike(self.state_dim, self.action_dim, filter_length, dropout, pretrained).float()
        self.net = self.net.to(device=self.device)
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay = 1e-5)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.loss_train = []
        self.loss_val = []
        self.acc_train = []
        self.acc_val = []
    
    def setup_wallet(self):
        self.wallet_log = f"backtests\\Log\\run_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}.csv"
        if not os.path.exists(self.wallet_log):
            with open(self.wallet_log, 'a') as file:
                (days_in_memory, lr, dropout, brain) = self.metadata
                file.write(f"#days in memory: {days_in_memory}\n")
                file.write(f"#lr: {lr}\n")
                file.write(f"#dropout: {dropout}\n")
                file.write(f"#brain: {brain}\n")
                file.write("#day\twallet\tacc\n")
        self.wallet = 1
        self.holding = {}
        for i in range(self.number_of_tickers):
            self.holding[i] = 0


    def reset_brain(self, dropout, learning_rate):
        self.setup_brain(self.filter_length, dropout, learning_rate, self.batch_size, self.epochs)

    def observe_past(self, day_open_position, day_close_position, Troubleshoot = False):
        batch = self.market.get_state_best_action(day_open_position, day_close_position, Troubleshoot)
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
    
    def learn_noval(self):
        train_loader = self.prepare_loader_noval()
        loss_train, acc_train = self.train_noval(train_loader)
        return loss_train, acc_train
    
    def learn(self, shuffle = True):
        train_loader, val_loader = self.prepare_loader(shuffle)
        loss_val, acc_val = self.train(train_loader, val_loader)
        return loss_val, acc_val
            
    @torch.no_grad()
    def predict(self, state, scale):
        self.net.train(False)
        prediction = self.net(state, scale)

        prediction = torch.argmax(torch.nn.Softmax(1)(prediction),1)[0]

        return prediction

    @torch.no_grad()
    def predict_batch(self, batch, check_pred = False):
        self.net.train(False)
        pairs, states, scales, best_actions = map(torch.stack, zip(*batch))

        states = states.float()
        states = states.to(device = self.device)
        scales = scales.float()
        scales = scales.to(device = self.device)
        best_actions = best_actions.type(torch.long).to(device = self.device).squeeze()
        prediction = self.net(states, scales)
        prediction = torch.argmax(torch.nn.Softmax(0)(prediction),1)

        acc = 0
        if check_pred:
            Ngood=len(prediction[prediction == best_actions])
            Nbad=len(prediction[prediction != best_actions])
            acc = Ngood/(Ngood+Nbad)

        return prediction, acc

    def act(self, arg_sorted_scores, day):
        price = self.market.market_price(day)

        for key in self.holding:
            self.wallet += self.holding[key] * price[key]
            self.holding[key] = 0

        Nstock = 1
        invest_val = self.wallet/(2*Nstock)
        for arg in arg_sorted_scores[:Nstock]:
            self.holding[arg] = - invest_val / price[arg]

        for arg in arg_sorted_scores[-Nstock:]:
            self.holding[arg] = invest_val / price[arg]


    def check_wallet(self, day):
        price = self.market.market_price(day)
        for key in self.holding:
            self.wallet += self.holding[key] * price[key]

    def prepare_loader_noval(self, shuffle = True):
        pairs, states, scales, best_actions = map(torch.stack, zip(*self.memory))

        train_ds = torch.utils.data.TensorDataset(states, scales, best_actions)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)

        return train_loader

    def prepare_loader(self, shuffle = True):
        pairs, states, scales, best_actions = map(torch.stack, zip(*self.memory))

        if shuffle:
            n = states.size(0)
            shuffle_ind = torch.randperm(n)

            states = states[shuffle_ind]
            scales = scales[shuffle_ind]
            best_actions = best_actions[shuffle_ind]

        cut_val = int(int(len(self.memory)) * 5/6)

        #here I fit on the last 5/6 and check on the first 1/6
        train_ds = torch.utils.data.TensorDataset(states[:cut_val], scales[:cut_val], best_actions[:cut_val])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        val_ds = torch.utils.data.TensorDataset(states[cut_val:], scales[cut_val:], best_actions[cut_val:])
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=True, num_workers=1)

        return train_loader, val_loader
    
    def train_one_epoch(self, train_loader):
        Ngood,Nbad = 0,0
        total_loss = 0
        total_acc = 0

        for i, data in enumerate(train_loader):
            state, scale, best_action = data
            state = state.float()
            state = state.to(device = self.device)
            scale = scale.float()
            scale = scale.to(device = self.device)

            best_action = best_action.type(torch.long).to(device = self.device).squeeze()

            self.optimizer.zero_grad()
            prediction = self.net(state, scale)

            loss = self.loss_fn(prediction, best_action)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            prediction = torch.argmax(torch.nn.Softmax(0)(prediction),1)

            Ngood+=len(prediction[prediction == best_action])
            Nbad+=len(prediction[prediction != best_action])

        total_acc = Ngood/(Ngood+Nbad)

        return total_loss/len(train_loader), total_acc

    @torch.no_grad()
    def check_val(self, val_loader):
        self.net.train(False)
        Ngood,Nbad = 0,0
        total_loss = 0
        total_acc = 0
        for i, data in enumerate(val_loader):
            state, scale, best_action = data
            state = state.float()
            state = state.to(device = self.device)
            scale = scale.float()
            scale = scale.to(device = self.device)
           
            best_action = best_action.type(torch.long).to(device = self.device).squeeze()

            prediction = self.net(state, scale)

            loss = self.loss_fn(prediction, best_action)
            
            total_loss += loss.item()

            prediction = torch.argmax(torch.nn.Softmax(0)(prediction),1)
            Ngood+=len(prediction[prediction == best_action])
            Nbad+=len(prediction[prediction != best_action])

        total_acc = Ngood/(Ngood+Nbad)
        
        return total_loss/len(val_loader) , total_acc

    def train(self, train_loader, val_loader):
        
        for epoch in range(self.epochs):
            self.net.train(True)
            loss_train, acc_train = self.train_one_epoch(train_loader)
            print(f"epoch {epoch}: train loss, acc: {np.round(loss_train,3)}, {np.round(acc_train,3)}")
            self.loss_train.append(loss_train)
            self.acc_train.append(acc_train)
    
            self.net.train(False)
            loss_val, acc_val = self.check_val(val_loader)
            print(f"epoch {epoch}: val    loss, acc: {np.round(loss_val,3)}, {np.round(acc_val,3)}")
            print("____")
            self.loss_val.append(loss_val)
            self.acc_val.append(acc_val)

        return loss_val, acc_val

    def train_noval(self, train_loader):
        
        for epoch in range(self.epochs):
            self.net.train(True)
            loss_train, acc_train = self.train_one_epoch(train_loader)
            print(f"epoch {epoch}: train loss, acc: {np.round(loss_train,3)}, {np.round(acc_train,3)}")
            self.loss_train.append(loss_train)
            self.acc_train.append(acc_train)

        return loss_train, acc_train

    def write(self, day, acc):
        with open(self.wallet_log, 'a') as file:
            file.write(f"{day}\t{self.wallet}\t{acc}\n")


    def save(self, save_path):
        torch.save(
            dict(model=self.net.state_dict()),
            save_path,
        )
        
        print(f"Net saved to {save_path}")

    def load(self, model):
        chkpt = torch.load(model)
        self.net.load_state_dict(chkpt['model'])