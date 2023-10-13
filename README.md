# Can a Convolutional Neural Network do Technical Analysis?

Technical Analysis to predict stock market moves has always looked like astrology to me.  
But TA has quite the large fan base, so there has to be some truth to it, even if by self realisation (if enough people believe that the "cup and handle" will give you a rise in price then you will at some point get some upward pressure).  
Even Burry ultimately relies on supports; in his own words:  
"With the market rallying since just prior to the start of the Strategy Lab, I must admit that many of the stocks I wanted to write about have already appreciated some.
This is problematic because even if I like a stock fundamentally, I am rarely willing to buy more than 15% above technical support.
I also generally use broken support as an exit point."

The idea is then simple, can I feed a few candle charts to a CNN and get has an output if I should buy now or not.  
As I want to have a market neutral strategy, let's reformulate:  
Can I feed the candle charts for 'stock0' and 'stock1' to a CNN and output 0 (1) if stock0 (stock1) outperforms?

## The idea

Let's start by chosing a sector, in our case, residential REIT stocks listed in the US.
We then select the 10 most traded stocks and download the historical data with Open, Close, High, Low and Volume (OCHLV) values for the one hour, one day, and one week candles ([data_extraction.py](data_extraction.py) helps with the task).

We can then translate these OCHLV data into visual candle charts, in the form of torch tensors, as in the figure below ( [you can plot your own](plot_candles.py)). 
And we can stack those tensors for a pair a stock.
Practically, we have the 1h, 1d and 1week andle charts for the pair of stocks so that we end up with 6 channels in pytorch nomenclature.
The height is fixed to 144 pixels (so that at least the height allows 4 2x2 maxpools and/or 2 3x3 ones).
The width will depend on the number of candles we want the CNN to look at.

<p align = "center">
<img src="images/all_candles.PNG" height=250>
<img src="images/tensor_stack.PNG" height=250>
</p>

We can note two things.  
First, if we have 10 stocks, we have 90 possible permutations which is good for training (100 days will give us 9000 tensors to train on).  
Second, the candles have to be scaled to fit in the 'pannel'. The scale is set by the one of the two stocks that varies the most on the periode we look at.
We need to give this scaling factor to the CNN to give it chance at infering the magnitude of the variation.

In a nutshell, the CNN take as input the stack of candles and the scaling values and output 0 or 1 depending on wether stock 0 or stock 1 is the one overperforming over the next 24h.  
And arbitrarily, we always open/close positions at 10:30am at the close of the first 1h candle of the day, this should be considered a parameter, but the parameter space is already enormous.

The achitecture of the network is shown below, you can also ave a look at the [code](neuralnet.py).

<p align = "center">
<img src="images\neuralnet.PNG" width = 700>
</p>


## How do we play?

Agent with memory and a neural net connected to a market that feeds him the candle charts.
Agent observe the stacks of the previous days and the variations of each stocks.
Memory of agent can have a finite size.
Agent learn from that.
Agent observe the 90 stacks of the day associated to the 10 stocks.
For each stacks we predict which one will be outperforming over one day.
Then we proceed to a vote. Each outpermorming of a pair get a point.
Once the 90 points are distributed we go long on the one with most points with half of the wallet and we open an equivalent short position on the one with the list point.
(We could go 100% of the wallet, but let's say we want to play it safe.)
Then we move to the next day, close both long and short positions and start again, pushing one more day in the memory.

## Some obvious flaws

Dividend
News
Order passed?
no fundamental
cost of transaction
market mood varies (the fear index) ... we will try to mitigate that one.

## Training and (hyper)parameters tuning

We can get 730days worth of 1h candle from the yfinance library.
Train and Validation set on first 200 days (a bit less than a year ~252days).
Cut in 5/6 1/6.
Shuffle or not => not Shuffling shows the issue with market mood.
Fine tune lr, dropout,n candles obsserved, filter lenght, holding time.
Pretrain the model.
Dropout might be too high, second model with lower one.

Then we play on 50 days (to add up to a year beacause why not).
Only train the last fully convoluted layer.
Fine tune, dropout, memory size, lr, epoch.

## Big leap, test with chosen parameters

MEH.
More tuning with longer holding time and different filter length?
How to get a CNN that is better at generalizing? Scale down or dramattically scale up?
Try to train without shuffle this time and fine tune with this. 
The output is too manichean, if the relative variation between the stocks is negligible (which should happen often?) training will force the NN to chose a side. This favorise fiiting situations that are not interresting.
Let's try to mitigate that.

## Trying to extract relative performance

Relative performance theoretically (but not practically) varies between -infty and +infty.
Transform this into 0,1.
tanh is a good candidate but what whould be the slope?
Naturally the CDF comes to mind.
Gives a optimal projected spacing on the 0,1 interval for all my states.
Indeed if too sharp, all 0 or all 1.
If not enough sharp, all projected on ~0.5.

Let's take the same NN, at the output, two values to which I take the softmax which gives a two values whose some = 1 and that belongs to [0,1]. We can look only at the first of those two.
My loss whould compare this value to the projected relative performance which is also between 0 and 1.
To have a good gradient for my backpropagation during training, I want optimally a loss that will gives me 0 when the softmax is equal to the relative variation and infinity when it's the distance between the two is the largest (i.e. 1).
ln(1-|softmax - projected relative variation|) has all the good properties. 