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


### The idea

Chose sector and the 10 most traded stocks (see blablabla.py for data extraction).
Transform the OCHL data into candles charts for 1h, 1d, 1wk (see wrapper) with n candles (parameter).
Stack a pair together (90 possible permutations => good for training).
The candles have to be scaled => scaling should also be fed to the CNN.
Feed them to a CNN (see below and neuralnet.py).
The CNN output 0 or 1.
I chose to open/close positions at 10:30am at the close of the first 1h candle of the day.

### How do we play?

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

### Some obvious flaws

Dividend
News
Order passed?
no fundamental
cost of transaction
market mood varies (the fear index) ... we will try to mitigate that one.

### Training and (hyper)parameters tuning

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

### Big leap, test with chosen parameters

MEH.
More tuning with longer holding time and different filter length?
How to get a CNN that is better at generalizing? Scale down or dramattically scale up?
Try to train without shuffle this time and fine tune with this. 
The output is too manichean, if the relative variation between the stocks is negligible (which should happen often?) training will force the NN to chose a side. This favorise fiiting situations that are not interresting.
Let's try to mitigate that.

### Trying to extract relative performance

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



This program is started from the command line by calling:

```
python flake-segmentation-tool.py --image_size image_size --image_folder images --output_folder output --num_classes num_classes
```

The `image_size` argument sets the width of the image (in pixels) when displayed in the GUI annotation window, `image_folder` is the path to the images to be annotated, `output_folder` is the path to save annotations and segmentations, and `num_classes` is the number of possible classes. Segmentations are saved as grayscale uint8 images with each pixel value proportional to the assigned class number. Starting the program opens four windows. The first image in the folder will be displayed in the annotation window:

<p align = "center">
<img src="screens/main_window.png" width=500>
</p>

Across the top there are buttons to increment through the images in the folder, and the "Go to Image" button loads an image selected by the nubmer in the the adjacent box. There are also options to save the labels and segmentations for a given image. The "Save Image" button allows the user to save the image with any transformations that have been applied. The class toggle buttons across the top indicate the presence or absence of the classes. Basic image-manipulation functions are controlled by the background tools window:

<p align = "center">
<img src="screens/background_tools.png" width=200>
</p>

"Background points" are added by the user by ctrl+clicking on the image. They need to be added in pairs defining lines. The pixels values along these lines are sampled and used to fit a quadratic illumnation profile common to microscope images; the illumnation profile is divided out by clicking "Subtract Background". Clicking white balance applies a factor to each channel so that the average color of the background points is (127, 127, 127). If no background points are selected, it sets the median color to (127, 127, 127). "Clip" clips all channel values at "Clip min" and "Clip max" and rescales the resulting values to lie between 0 and 255.

The segmentation tools windows has options for auto-segmentation:
<p align = "center">
<img src="screens/segmentation_tools.png" width=200>
</p>

The basic method for segmentation is user defined polygons (vertices created by left clicking). The polygons are completed by clicking "Close Poly". The "Active Class" box determines the class of any segmentations. In addition, there are several "hyperparameters" that control what happens when the user clicks "Auto Segmentation". The "Filter Size" parameter controls the size of a box averaging filter applied to the image before any other auto-segmentation operations. The simplest auto-segmentation operation is to send all pixels with color values in a selected range to the active class. This range can be entered directly in the "Color range min" and "Color range max" boxes, or automatically determined using "Dropper points". To use dropper points, shift+right click on the desired regions, then press "Color range from selection". Checking the "Color range to class" box then clicking "Auto Segmentation" creates the segmentation:

<p align = "center">
<img src="screens/segmentation_1.png" width=500>
</p>

The is also a fuzzy select algorithm using flood fill. The fill points are selected by shift-clicking. The tolerance parameter determines how large the difference bewtween adjacent pixels needs to be to stop the flood. Clicking "Auto Segmentation" applies both the color-range-to-class and any flood fills for the active class:

<p align = "center">
<img src="screens/segmentation_2.png" width=500>
</p>

There a few more parameters that we can use to control the auto-segmentation. "Class pooling size" controls the size of a square max-pooling element applied to the segmentation. This is applied to the color-range-to-class and flood fill segmentation masks, but not the polygons. Checking "Background exclusion" modifies the flood fill algorithm by attempting to further exclude background regions from the flood (beyond what is achieved by the threshold parameter). It essentially prevents the flood from entering any region too close in color to the background color (the required difference is set by "Exclusion tolerance"). This allows for larger values of flood tolerance parameter and helps in generating smooth fills with a small number of flood points. 

Finally, the "Key Bindings" window shows a list of hotkeys that can be used to operate the segementation tool.