from agent import Trader
from matplotlib import pyplot as plt
import torch 
import numpy as np
import random
import time

def loguniform(low=0, high=1):
    return np.power(10, np.random.uniform(low, high))

if __name__ == "__main__":
    log = r"C:\Users\Dodo\Documents\git\MarketNeutral_CNN\hyperparam_opti\candles.csv"

    with open(log, 'w') as file:
        file.write("#number of candles\thold time\tdoprout\tlr\ttrain loss\ttrain acc\tval loss\tval acc\n")


    N = 1

    days_in_memory = 200
    filter_length = 3
    batch_size = 16
    epochs = 20
    holding_times = [1]
    candles = [3,7,10,14,20,30,40,50]

    for i in range(len(candles)):
        observed_candles = random.randint(8,60)
        observed_candles = candles[i]
        holding_time = random.choice(holding_times)
        lr = 5e-5
        dropout = 0.55

        Mario = Trader(holding_time, observed_candles, days_in_memory, filter_length, dropout, lr, batch_size, epochs)
        days = Mario.market.get_available_days()
        Ndays_skip = int(observed_candles / 7) + 1
        print(f"skip {Ndays_skip} days to have at least {observed_candles} 1h candles in the past")

        Ndays = 200

        counter = 0

        t0 = time.time()
        for i in range(Ndays_skip, Ndays_skip + Ndays):
            counter += 1
            print(counter)

            #Observe
            day_batch = Mario.observe_past(days[i], days[i+holding_time])
            Mario.memorize(day_batch)
        print("Time to aggregate 200 days:", time.time()-t0)

        for j in range(N):
            # dropout = np.random.uniform(0.15,0.5)
            # lr = loguniform(-5.5,-4)
            # lr = 1e-5
            
            print("####")
            print("number of candles:", observed_candles)
            print("holding time:", holding_time)
            print("lr:", lr)
            print("dropout:", dropout)
            print("####")

            Mario.reset_brain(dropout,lr)

            t0 = time.time()
            Mario.learn()
            print("Time to train",time.time()-t0)
            
            with open(log, 'a') as file:
                file.write(f"{observed_candles}\t{holding_time}\t{dropout}\t{lr}\t{Mario.loss_train[-1]}\t{Mario.acc_train[-1]}\t{Mario.loss_val[-1]}\t{Mario.acc_val[-1]}\n")
            


