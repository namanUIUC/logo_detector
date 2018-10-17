%% DriveInformed logo Detection

% Using SIFT from online
% ---------------------------------------------------------------
%                LOWE'S SCALE INVARIANT FEATURE TRANSFORM
% 
%                             Andrea Vedaldi
%                          vedaldi@cs.ucla.edu
%                    http://www.cs.ucla.edu/~vedaldi/
% CREDITS
%   The SIFT algorithm [1] has been patented by David Lowe. Some of the
%   images in 'data/' are from [2,3].
% 
% [1] D. G. Lowe, "Distinctive image features from scale-invariant
%     keypoints," IJCV, vol. 2, no. 60, pp. 91 110, 2004.
% 
% [2] K. Mikolajczyk, T. Tuytelaars, C. Schmid, A. Zisserman, J. Matas,
%     F. Schaffalitzky, T. Kadir, and L. Van Gool, "A comparison of affine
%     region detectors," IJCV, vol. 1, no. 60, pp. 63 86, 2004.
% 
% [3] C. Hormann, "Landscape of the week 2," 2006.
clear all;clc;
addpath('SIFTlib')
checkCompilation();

% Hyperparameters
dthr        = 0.4;
alpha       = 1.2;
sigma       = 4;
threshold   = 0.08;

% Maps for the labels
ids = [1 2 3 4 5];
names = {'Bank of America','Capital One','Citi Group', 'JP Morgan Chase', 'Wells Fargo'};
Map = containers.Map(ids,names);

% Feature Extraction for Templates
fprintf('Extracting templates SIFT features...')
logoImages=extractFeaturesBulk('./Data/logos/');
fprintf('done\n')

% Feature Extraction for Test Image
fprintf('Extracting test image SIFT features...')
testImages=extractFeaturesBulk('./Data/test/');
fprintf('done\n')

% Prediction
fprintf('Prediction:')
tic

% Procedure to match features between image and logo
% Matching is done using exhaustive search. Using LSH methods
% the process time is reduced about 10 times.

for i=1:length(testImages)
    temp=zeros(1,length(logoImages));
    for j=1:length(logoImages)
        
        pairs = matchFeatures(testImages(i),logoImages(j),dthr);
        % Fast Geometric Consistency Test (FGCT) logic
        temp(1,j) = FGCT(testImages(i),logoImages(j),pairs,alpha,sigma);
    end
    correspondance(i,:) = temp;
    
    % Classification
    x = temp;
    x(temp < threshold) = 0;
    [value, indx] = max(x);
    if (value < threshold)
        fprintf('\nThe Image %d class is : %s\n', i,'Other')
    else
        fprintf('\nThe Image %d class is : %s\n', i, Map(indx))
    end
end
d = toc;
fprintf('done\n')

% Execution Details
disp(['Average classification time: ' num2str(1000*d/(numel(correspondance))) 'ms per image'])
