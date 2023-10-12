# Market Neutral Investment base on CNN

Brief description.
Techniacal Analysis always looks like astrology.
Might be some truth to it, if only via self realisation.
Also Burry said: ...

Why not give some statistical analysis of it through Conv Neural Network?
Idea is then the following:
    Create plates showing candles for the last x hours, last x days and last x weeks.
    Give this plates for two stocks to a convulotional neural network
    The CNN will give as an output which stock is the most favorable to buy

Some notes on the fear and greed index.

### Requirements

NumPy 1.15.4, Pillow 5.3.0, PyQt 5.9.2, SciPy 1.1.0, scikit-image 0.16.2

### Usage

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