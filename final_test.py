from agent import Trader
from matplotlib import pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    brain_Luigi = "Net/Net11/full_shuffle_epoch25.chkpt"
    brain_Mario = "Net/Net22/full_shuffle_epoch25.chkpt"
    
    number_of_days_play = 450

    filter_length = 3
    batch_size = 16
    epochs = 10
    holding_time = 1
    observed_candles = 7
    days_in_memory = 15
    lr = 1.5e-3
    dropout_Luigi = 0.55
    dropout_Mario = 0.35

    day0 = 255

    Mario = Trader(holding_time, observed_candles, days_in_memory, filter_length, dropout_Mario, lr, batch_size, epochs, pretrained = True, metadata = (days_in_memory, lr, dropout_Mario, brain_Mario))
    Mario.load(brain_Mario)
    wallet_Mario = [Mario.wallet]

    time.sleep(5) #to allow log to change name
    Luigi = Trader(holding_time, observed_candles, days_in_memory, filter_length, dropout_Luigi, lr, batch_size, epochs, pretrained = True, metadata = (days_in_memory, lr, dropout_Luigi, brain_Luigi))
    Luigi.load(brain_Luigi)
    wallet_Luigi = [Luigi.wallet]

    time.sleep(5) #to allow log to change name
    Concensus = Trader(holding_time, observed_candles, days_in_memory, filter_length, dropout_Luigi, lr, batch_size, epochs, pretrained = True, metadata = (days_in_memory, lr, dropout_Luigi, brain_Luigi))
    wallet_Concensus = [Concensus.wallet]

    x = [day0 - 1 ]

    #get the days
    days = Mario.market.get_available_days()

    #prepare plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line_Mario, = ax.plot(x,wallet_Mario, 'r-')
    line_Luigi, = ax.plot(x,wallet_Luigi, 'g-')
    line_Concensus, = ax.plot(x,wallet_Concensus, 'k-')

    #start playing
    Accuracy_Mario = []
    Accuracy_Luigi = []
    for i in range(day0 - days_in_memory, day0 - days_in_memory + number_of_days_play):

        yesterday = days[i]
        today = days[i+holding_time]
        tomorrow = days[i+holding_time+1]

        print(days[i])
        print("wallet Mario: ", wallet_Mario[-1])
        print("wallet Luigi: ", wallet_Luigi[-1])
        print("wallet Concensus: ", wallet_Concensus[-1])

        if len(Mario.loss_train)>0:
            print("loss Mario: ", Mario.loss_train[-1])
            print("loss Luigi: ", Luigi.loss_train[-1])
        print("-----")

        #Observe the last possible day where we know the best action and memorize it.
        past_batch = Mario.observe_past(yesterday, today)
        Mario.memorize(past_batch)
        Luigi.memorize(past_batch)

        if len(Mario.memory)/len(past_batch) < days_in_memory:
            continue
        
        #Learn
        loss_train_Mario, acc_train_Mario = Mario.learn_noval()
        loss_train_Luigi, acc_train_Luigi = Luigi.learn_noval()
    
        #Observe, Predict, and Act for today
        day_batch = Mario.observe_past(today, tomorrow)
        scores_Mario = np.zeros(Mario.number_of_tickers)
        scores_Luigi = np.zeros(Luigi.number_of_tickers)
            
        prediction_Mario, acc_Mario = Mario.predict_batch(day_batch, check_pred = True)
        Accuracy_Mario.append(acc_Mario)
        prediction_Luigi, acc_Luigi = Luigi.predict_batch(day_batch, check_pred = True)
        Accuracy_Luigi.append(acc_Luigi)
        print(f"accuracy Mario = {np.round(acc_Mario*100,2)}%")
        print(f"accuracy Luigi = {np.round(acc_Luigi*100,2)}%")
        
        for j in range(len(prediction_Mario)):
            pair = day_batch[j][0]
            if prediction_Mario[j] == 0:
                scores_Mario[pair[0]] += 1
            else:
                scores_Mario[pair[1]] += 1
            
            if prediction_Luigi[j] == 0:
                scores_Luigi[pair[0]] += 1
            else:
                scores_Luigi[pair[1]] += 1

        scores_Concensus = scores_Mario + scores_Luigi

        arg_scores_sorted_Mario = np.argsort(scores_Mario)
        arg_scores_sorted_Luigi = np.argsort(scores_Luigi)
        arg_scores_sorted_Concensus = np.argsort(scores_Concensus)

        Mario_buys = Mario.market.tickers_list[arg_scores_sorted_Mario[-1]]
        Mario_sells = Mario.market.tickers_list[arg_scores_sorted_Mario[0]]
        Luigi_buys = Luigi.market.tickers_list[arg_scores_sorted_Luigi[-1]]
        Luigi_sells = Luigi.market.tickers_list[arg_scores_sorted_Luigi[0]]
        Concensus_buys = Concensus.market.tickers_list[arg_scores_sorted_Concensus[-1]]
        Concensus_sells = Concensus.market.tickers_list[arg_scores_sorted_Concensus[0]]

        print(f"Mario buys {Mario_buys} and sells {Mario_sells}")
        print(f"Luigi buys {Luigi_buys} and sells {Luigi_sells}")
        print(f"Concensus buys {Concensus_buys} and sells {Concensus_sells}")

        Mario.act(arg_scores_sorted_Mario, today)
        Luigi.act(arg_scores_sorted_Luigi, today)
        Concensus.act(arg_scores_sorted_Concensus, today)

        #Log
        wallet_Mario.append(Mario.wallet)
        wallet_Luigi.append(Luigi.wallet)
        wallet_Concensus.append(Concensus.wallet)
        x.append(i)


        #Write
        Mario.write(today, acc_Mario)
        Luigi.write(today, acc_Luigi)
        Concensus.write(today, (acc_Luigi+acc_Mario)/2)

        line_Mario.set_data(x,wallet_Mario)
        line_Luigi.set_data(x,wallet_Luigi)
        line_Concensus.set_data(x,wallet_Concensus)

        miny = min(min(wallet_Mario),min(wallet_Luigi),min(wallet_Concensus))*0.95
        maxy = max(max(wallet_Mario),max(wallet_Luigi),max(wallet_Concensus))*1.05

        ax.axis([min(x),max(x)*1.05,miny,maxy])

        fig.canvas.draw()
        fig.canvas.flush_events()

        if i > day0 - days_in_memory:
            Mario.epochs = 2
            Luigi.epochs = 2