# Implementation Version 3

## Improved Version 

Since, the last version that is based on bags of descriptors is not performing well, I need to take another approach to tackle this issue. In this imporved implementation I have used another technique that is very frequently used in logo detection. This technique is called **Fast Geometric Consistency Test (FGCT)**. FGCT is used for logos/trademarks or object detection and clasification from test images

## Implementation:

I have used `MATLAB` for this implementation as this requires `SIFT` feature extractor which is not available in the `opencv` versions due to patent issues. I have used SIFTlib available online to complie in the `MATLAB` to implement **Fast Geometric Consistency Test (FGCT)**. This method  is done in three steps:

- Extract features (SIFT) from test and reference logo image.
- Match test image features with logo images feature in the descriptor space.
- Use matched pair and using FGCT calculate the corresponding features that forms a consistent geometry on image and logo feature sets.

For classification :

- Threshold the correspondences with some value so that we could be certain about the classes and otherwise classify as `other` class.
- Finally mapping is done with maximum argument (if exist) to the labels.

The original paper is here : N. Zikos and A. Delopoulos, "Fast geometric consistency test for real  time logo detection," Content-Based Multimedia Indexing (CBMI), 2015  13th International Workshop on, Prague, 2015, pp. 1-6. doi: 10.1109/CBMI.2015.7153636



## Results on validation set

```
Extracting test image SIFT features...done
Extracting logo reference image SIFT features...done
Calculating correspondances for every test image for every logo...
The Image 1 class is : Bank of America

The Image 2 class is : Other (False)

The Image 3 class is : Citi Group

The Image 4 class is : Other

The Image 5 class is : Other

The Image 6 class is : JP Morgan Chase

The Image 7 class is : Wells Fargo
done
Average execution time: 33.6044ms per image
```

This method has performed significantly better that previous one. With 6 / 7 validation images to be correctly identified. Although, more data is needed to be fully confident on the model.

## Running the project

To run the project the you will need MATLAB 2017b. 

```
Steps to run the code:
	1. Place the logos in the Data/logos folder so it could be used as templates.
	2. Place the test images in the Data/test folder
	3. execute main.m
	4. The predictions will be there in the console. 
	5. Check for correspondence values for confidence levels w.r.t all the classes
```

## General Comments

The state of the art models are based on deep nets implementations. The reason I cant use deep nets for this implementation is that deep neural networks consumes great amount of data. And since, only few images were provided to me, I am restricted to implement classical techniques for the the detection purpose. Provided we have significant number of images, the detection accuracy can go way beyond accuracy achieved via classical algorithms.  

# References : 

1. https://ieeexplore.ieee.org/document/7153636
2. SIFT lib - vedaldi@cs.ucla.edu
3. D. G. Lowe, "Distinctive image features from scale-invariant keypoints," IJCV, vol. 2, no. 60, pp. 91 110, 2004.
4. K. Mikolajczyk, T. Tuytelaars, C. Schmid, A. Zisserman, J. Matas, F. Schaffalitzky, T. Kadir, and L. Van Gool, "A comparison of affine region detectors," IJCV, vol. 1, no. 60, pp. 63 86, 2004.
5. C. Hormann, "Landscape of the week 2," 2006.