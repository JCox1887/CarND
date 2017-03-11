#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the image to gray scale, then I used Canny edge detection with a mask to so that when I pass the result into my hough lines function with the draw_lines function built in, I used the draw_lines function to connect the individual ilines to trace the detected lanes 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by using an if statement that tested for the the slope of a line to be positive or negative, once this was decided I extended the line using the difference between the total height of the image less the y coordinate and divided by the slope.  Then the function averages the lines to the side of the main one to ensure they trend together.





###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be it is too decided for a best case senario it has not been tested with night time images or bad weather.



###3. Suggest possible improvements to your pipeline

A possible improvement would be to test it on different senarios and ensure that the edge detection still works as expected. One way to fix this could be to find other other color spaces that may be more precise and makes the pipeline more consistent.

