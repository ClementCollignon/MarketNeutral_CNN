from agent_variation import Trader
from matplotlib import pyplot as plt
import numpy as np
import time
from collections import deque

if __name__ == "__main__":
    brain = "Nets/Net_var1/full_shuffle_epoch852.chkpt"
    
    number_of_days_play = 1000

    filter_length = 3
    batch_size = 90
    epochs = 1
    holding_time = 1
    observed_candles = 7
    days_in_memory = 1
    lr = 1e-7
    dropout = 0.4
    threshold = 0/100
    slope = 8

    day0 = 253

    Mario = Trader(slope,  holding_time, observed_candles, days_in_memory, filter_length, dropout, lr, batch_size, epochs, frozen = False, metadata = (days_in_memory, lr, dropout, brain, threshold))
    Mario.load(brain)
    wallet = [Mario.wallet]

    x = [day0 - 1]

    success = deque(maxlen = 10)
    polarity = 1

    #get the days
    days = Mario.market.get_available_days()

    #prepare plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line_Mario, = ax.plot(x,wallet, 'k-', lw = 3)
    
    #start playing
    Dist_Mario = []
    for i in range(day0 - days_in_memory, day0 - days_in_memory + number_of_days_play):

        yesterday = days[i]
        today = days[i+holding_time]
        tomorrow = days[i+holding_time+1]

        print(days[i])
        print("wallet Mario: ", wallet[-1])

        if len(Mario.dist_train)>0:
            print("error Mario: ", Mario.dist_train[-1])
        print("-----")

        #Observe the last possible day where we know the best action and memorize it.
        past_batch = Mario.observe_past(yesterday, today)
        Mario.memorize(past_batch)

        if len(Mario.memory)/len(past_batch) < days_in_memory:
            continue
        
        #Learn
        # Mario.prepare_loader_noval()
        # loss_train_Mario, dist_train_Mario = Mario.learn_noval()
    
        #Observe, Predict, and Act for today
        day_batch = Mario.observe_past(today, tomorrow)
        scores = np.zeros(Mario.number_of_tickers)
            
        prediction, dist = Mario.predict_batch(day_batch, check_pred = True)
        print(f"mean distance to truth = {np.round(dist*100,2)}%")
        
        for j in range(len(prediction)):
            pair = day_batch[j][0]
            scores[pair[0]] -= prediction[j] #careful here, the relative variation is variation stock1 - variation stock 0
            scores[pair[1]] += prediction[j]

        if np.sum(np.array(success))<0 and len(success) == success.maxlen:
            polarity = -1 * polarity
            for j in range(len(success)):
                success[j] = - success[j]

        Mario.act(scores, today, threshold, polarity)

        #Log
        wallet.append(Mario.wallet)
        x.append(i)

        gain = (wallet[-1] - wallet[-2]) / wallet[-2]
        success.append(gain)
        
        print(polarity)
        print(success, np.sum(np.array(success)))

        #Write
        Mario.write(today, dist)

        line_Mario.set_data(x,wallet)
        
        miny = min(wallet)*0.95
        maxy = max(wallet)*1.05

        ax.axis([min(x),max(x)*1.05,miny,maxy])

        fig.canvas.draw()
        fig.canvas.flush_events()

        if i > day0 - days_in_memory:
            Mario.epochs = 1