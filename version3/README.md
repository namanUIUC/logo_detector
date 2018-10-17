# Implementation Version 3

## Overall Idea of implementation

According to the problem statement, We need to identify (detect) the logo based on an input image. 

### About the dataset

We are given a [dataset](./training_data/) of 12 images. These images belong to the following classes : `Bank of America`, `Capital One`, `Citigroup`, `JPMorganChase`, `WellsFargo` and `Others`. Additionally we are given few sample images to do the model validation. These sample images consists of `headers` and `logos(from different classes)` 

## Implementation:

The most obvious method of implementation is by using [template matching](https://www.wiley.com/en-us/Template+Matching+Techniques+in+Computer+Vision%3A+Theory+and+Practice-p-9780470517062). To find the implementation please refer [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html). But unfortunately, this naive technique have some implementation issues as far as logo detection is concerned.  The major issue is that if the test image is scaled, then the template will not be able to detect the exact position of the logo itself. This would make the model unstable under the scale variance. 

The other technique that we can use is [matching descriptors](https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html). In this implementation we can extract the descriptor vectors of the image which are invariant to scaling and intensity. This would make our model robust to scale and intensity variation. 

For this version I have chosen BLOB DETECTOR which uses corner detection technique (Harris corner detector) for the descriptor and key-points extraction. The blob detectors detect the image corners and represent as blobs which plays an important role in tracking and object detection.  Below are the steps involved in the implementation :

1. Some hyper parameters are necessary to define for a single implementation. Here, initial sigma value for gaussian function used for filtering in laplacian of gaussian, number of step in the space scale and the threshold values for maximum suppression are those hyper parameters. Each combination of these parameters would result in different outcome and can be tuned according to user preferences.

2. The input image provided is raw and need to preprocess. The image is first converted to grayscale and then converted its data type to double followed by normalization.

3. After preprocessing, the blob detection algorithm is applied as follows:

   1. Scaling the filter operation (inefficient):

      ![Image result for scale invariant](https://docs.opencv.org/3.0-beta/_images/sift_dog.jpg)

   2. Convolve the image with filter of laplacian of gaussian with given sigma value. For this process MATLAB fspecial function is used (with keeping an odd filter as a constraint via sigma value) to generate filter.

   3. Compare each pixel in the convolved image with a set of neighbourhood and selection of the best is performed. 
      Repeat the above 2 step to create space scale (with hyperparameter of steps) with different values of sigma obtained by factor multiplication of K and thereby generating different size of filter to convolve an image with. As, we are only concerned with the maximum response of a pixel to LOG filter, non- maximum suppression is performed in 2D slices.
      Maximum value across the scale space is then taken as non-maximum suppression in 3D. 
      Now we consider only those pixel which passes survival threshold. The threshold value is the same value form hyperparameter. Then the characteristic radius and the center is calculated with the following equation:  r = σ. √2 

   4. Scaling the image operation (efficient):

      Given an image and a sigma value, the laplacian filter is created to convolve with the image.
      The given image is then resized with factor of inverse of k .

      ![Related image](http://campar.in.tum.de/twiki/pub/Chair/KlinkerCMU/FidoWarpPyramid.JPG)

      The similar operation of the LoG convolution is applied to the resized image. The image is then rescaled again to normal dimensions. 
      Similar steps starting from 2D non maximum suppression is applied to this procedure as well.



### Results:

Here's how the filter looks like:

![filter](./output/Process/filter.gif)

Below are the images snaps taken on different level of filter. This makes the processes scale invariant. One of the sample image of `Bank of America` is taken here:

![sample](./output/sample/sample.gif)

Here is one of the output of the BLOB DETECTORS:

![final_blob](./output/final_blob.jpg)

## Bag of Descriptors:

> **NOTE**:  The above implementation is done on MATLAB. Please refer to the repository for codes.

Now that we have extracted the features that invariant to rotation as well as as scaling, we can collect all these descriptors `Bag of Descriptors`and then compare the testing image via brute force or RANSAC.

The further implementation is done in version 2 of the project. 



## Running the project

To run the project the you will need MATLAB 2017b. 

```
Steps to run the code:
	1. Place the images in the data folder.
	2. Replace the variable name in the main.m
	3. execute main.m
```

# References : 

1. https://www.wiley.com/en-us/Template+Matching+Techniques+in+Computer+Vision%3A+Theory+and+Practice-p-9780470517062
2. https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
3. https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html
4. David G Lowe. Distinctive image features from scale-invariant keypoints.
5. Sample Harris detector code