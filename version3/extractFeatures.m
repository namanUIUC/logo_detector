function features=extractFeatures(filename)

I = imread(filename);
if size(I,3)==3
    I=rgb2gray(I);
end
I=double(I)/256;
[features.frames,features.descriptors] = sift(I, 'Verbosity', 0);
features.file = filename;
features.numFeatures = size(features.frames,2);
