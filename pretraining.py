from agent import Trader
from matplotlib import pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    log = r"hyperparam_opti\trainingnoshuffle55.csv"
    path_NN = "Nets/Net1"
    days_in_memory = 200
    filter_length = 3
    batch_size = 64
    epochs = 1
    holding_time = 1
    observed_candles = 7
    lr = 1e-4
    dropout = 0.55
    brain = "Nets/Net1/full_shuffle_epoch99.chkpt"

    # with open(log, 'w') as file:
    #     file.write("#number of candles\thold time\tdoprout\tlr\ttrain loss\ttrain acc\tval loss\tval acc\n")


    metadata = (days_in_memory, lr, dropout, brain)

    Mario = Trader(holding_time, observed_candles, days_in_memory, filter_length, dropout, lr, batch_size, epochs, metadata = metadata)
    Mario.load(brain)
    days = Mario.market.get_available_days()
    Ndays_skip = int(observed_candles / 7) + 1
    print(f"skip {Ndays_skip} days to have at least {observed_candles} 1h candles in the past")

    Ndays = 250

    counter = 0
    t0 = time.time()
    for i in range(Ndays_skip, Ndays_skip + Ndays):
        counter += 1
        print(counter)

        #Observe
        day_batch = Mario.observe_past(days[i], days[i+holding_time])
        Mario.memorize(day_batch)
    print(f"Time to aggregate {Ndays} days:", time.time()-t0)

    Mario.prepare_loader(shuffle = True)
    for i in range(100):
        Mario.learn()
        path = f"{path_NN}/full_shuffle_epoch{i}.chkpt"
        Mario.save(path)
        print("Time to train",time.time()-t0)
        with open(log, 'a') as file:
            file.write(f"{observed_candles}\t{holding_time}\t{dropout}\t{lr}\t{Mario.loss_train[-1]}\t{Mario.acc_train[-1]}\t{Mario.loss_val[-1]}\t{Mario.acc_val[-1]}\n")
    
   
        


