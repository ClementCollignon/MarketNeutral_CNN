from agent import Trader
from matplotlib import pyplot as plt
import torch 
import numpy as np
import random


def loguniform(low=0, high=1):
    return np.power(10, np.random.uniform(low, high))

def run(number_of_days_play, days_in_memory, lr, dropout, brain):
    filter_length = 3
    batch_size = 16
    epochs = 10
    holding_time = 1
    observed_candles = 7

    print("#####")
    print("memory:", days_in_memory)
    print("lr:", lr)
    print("dropout:", dropout)
    print("brain:", brain)

    Mario = Trader(holding_time, observed_candles, days_in_memory, filter_length, dropout, lr, batch_size, epochs, pretrained = True, metadata = (days_in_memory, lr, dropout, brain))
    Mario.load(brain)
    wallet = [Mario.wallet]
    x = [200 + days_in_memory]

    #get the days
    days = Mario.market.get_available_days()

    #prepare plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x,wallet, '-')

    #start playing
    Accuracy = []
    for i in range(201, 201 + number_of_days_play + days_in_memory):

        yesterday = days[i]
        today = days[i+holding_time]
        tomorrow = days[i+holding_time+1]

        print(days[i])
        print("wallet: ", wallet[-1])

        if len(Mario.loss_train)>0:
            print("loss: ", Mario.loss_train[-1])
        print("-----")

        #Observe the last possible day where we know the best action and memorize it.
        past_batch = Mario.observe_past(yesterday, today)
        Mario.memorize(past_batch)

        if len(Mario.memory)/len(past_batch) < days_in_memory:
            continue
        
        #Learn
        loss_train, acc_train = Mario.learn_noval()
    
        #Observe, Predict, and Act for today
        day_batch = Mario.observe_past(today, tomorrow)
        scores = np.zeros(Mario.number_of_tickers)
            
        prediction, acc = Mario.predict_batch(day_batch, check_pred = True)
        Accuracy.append(acc)
        print(f"accuracy = {np.round(acc*100,2)}%")
        
        for j in range(len(prediction)):
            pair = day_batch[j][0]
            if prediction[j] == 0:
                scores[pair[0]] += 1
            else:
                scores[pair[1]] += 1

        print(np.sort(scores))
        arg_scores_sorted = np.argsort(scores)
        Mario.act(arg_scores_sorted, today)

        #Log
        wallet.append(Mario.wallet)
        x.append(i)

        #Write
        Mario.write(acc)

        line1.set_data(x,wallet)
        ax.axis([min(x),max(x)*1.05,min(wallet)*0.95,max(wallet)*1.05])

        fig.canvas.draw()
        fig.canvas.flush_events()

        if i > 201:
            Mario.epochs = 2

if __name__ == "__main__":
    brains = ["Net2/full_shuffle_epoch99.chkpt","Net1/full_shuffle_epoch99.chkpt"]
    number_of_days_play = 50
    
    N = 1
    dropouts = np.linspace(0.2,0.8,N)

    for i in range(N):
        # days_in_memory = random.randint(2,30)
        days_in_memory = 15
        lr = 1.5e-3
        # dropout = random.uniform(0.2,0.8)
        # dropout = dropouts[i]
        dropout = 0.55
        # brain = random.choice(brains)
        
        # brain = brains[0]
        # run(number_of_days_play, days_in_memory, lr, dropout, brain)
        # plt.close('all')

        brain = brains[1]
        run(number_of_days_play, days_in_memory, lr, dropout, brain)
        plt.close('all')