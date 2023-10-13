from agent_variation import Trader
from matplotlib import pyplot as plt
import random
import numpy as np
import time
    
def loguniform(low=0, high=1):
    return np.power(10, np.random.uniform(low, high))

if __name__ == "__main__":
    log = r"C:\Users\Dodo\Documents\git\MarketNeutral_CNN\hyperparam_opti_variation\rough.csv"

    # with open(log, 'w') as file:
    #     file.write("#number of candles\thold time\tdoprout\tlr\tslope\ttrain loss\ttrain dist\tval loss\tval dist\n")

    epochs = 20
    batch_size = 64
    holding_time = 1 #day
    number_of_observed_candles = 8
    days_in_memory = 250
    filter_length = 3
    N = 100

    for i in range(1):
        dropout = 0.20
        learning_rate = loguniform(-5,-3)
        slope = random.uniform(1,20)

        Mario = Trader(slope, holding_time, number_of_observed_candles, days_in_memory, filter_length, dropout, learning_rate, batch_size, epochs, frozen = False, metadata = None)
        days = Mario.market.get_available_days()
        Ndays_skip = int(number_of_observed_candles / 7) + 1
        print(f"skip {Ndays_skip} days to have at least {number_of_observed_candles} 1h candles in the past")

        Ndays = 50
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
            dropout = random.uniform(0.1,0.5)
            learning_rate = loguniform(-5,-3)
            slope = random.uniform(1,20)
            
            print("####")
            print("lr:", learning_rate)
            print("dropout:", dropout)
            print("slope:", slope)
            print("####")

            Mario.reset_brain(dropout,learning_rate)

            t0 = time.time()
            Mario.learn()
            print("Time to train",time.time()-t0)
            
            with open(log, 'a') as file:
                file.write(f"{number_of_observed_candles}\t{holding_time}\t{dropout}\t{learning_rate}\t{slope}\t{Mario.loss_train[-1]}\t{Mario.dist_train[-1]}\t{Mario.loss_val[-1]}\t{Mario.dist_val[-1]}\n")
    
    