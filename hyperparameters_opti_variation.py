from agent_variation import Trader
from matplotlib import pyplot as plt
import random
import numpy as np
import time
    
def loguniform(low=0, high=1):
    return np.power(10, np.random.uniform(low, high))

if __name__ == "__main__":
    log = "hyperparam_opti_variation/fine_3.csv"

    epochs = 20
    batch_size = 90
    holding_time = 1 #day

    filter_lengths = [9,11,13]

    
    N = 20

    for i in range(N):
        number_of_observed_candles_arr = [random.randint(30,35), random.randint(7,9)]
        number_of_observed_candles = random.choice(number_of_observed_candles_arr)
        days_in_memory = 250 #doesn't matter here

        filter_length = 3 #doesn't matter here will be setup later randomly
        dropout = 0.20 #doesn't matter here will be setup later randomly
        learning_rate = 1e-4 #doesn't matter here will be setup later randomly
        slope = 1 #doesn't matter here will be setup later randomly

        Mario = Trader(slope, holding_time, number_of_observed_candles, days_in_memory, filter_length, dropout, learning_rate, batch_size, epochs, frozen = False, metadata = (0, 0, 0, "",0))
        days = Mario.market.get_available_days()
        Ndays_skip = int(number_of_observed_candles / 7) + 1
        Ndays_skip = int(40 / 7) + 1
        print(f"skip {Ndays_skip} days to have at least {number_of_observed_candles} 1h candles in the past")

        Ndays = 200
        counter = 0

        t0 = time.time()
        for i in range(Ndays_skip, Ndays_skip + Ndays):
            counter += 1
            print(counter)

            #Observe
            day_batch = Mario.observe_past(days[i], days[i+holding_time])
            Mario.memorize(day_batch)
        print("Time to aggregate 100 days:", time.time()-t0)

        for j in range(N):
            dropout = random.uniform(0.2,0.9)
            learning_rate = 5e-7
            slope = random.uniform(20,40)
            filter_length = random.choice(filter_lengths)
            
            print("####")
            print("number of observed candles:", number_of_observed_candles)
            print("filter length:", filter_length)
            print("lr:", learning_rate)
            print("dropout:", dropout)
            print("slope:", slope)
            print("####")

            Mario.reset_brain(dropout,learning_rate, slope, filter_length)

            t0 = time.time()
            Mario.prepare_loader(shuffle=False)
            Mario.learn()
            print("Time to train",time.time()-t0)
            
            with open(log, 'a') as file:
                file.write(f"{number_of_observed_candles}\t{filter_length}\t{holding_time}\t{dropout}\t{learning_rate}\t{slope}\t{np.min(Mario.loss_train)}\t{np.min(Mario.dist_train)}\t{np.min(Mario.loss_val)}\t{np.min(Mario.dist_val)}\n")
    
    