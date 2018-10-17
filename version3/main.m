%% D
clear all;clc;
addpath('SIFTlib')
checkCompilation();

ids = [1 2 3 4 5];
names = {'Bank of America','Capital One','Citi Group', 'JP Morgan Chase', 'Wells Fargo'};
Map = containers.Map(ids,names);

fprintf('Extracting test image SIFT features...')
testImages=extractFeaturesBulk('./Data/test/');
fprintf('done\n')
fprintf('Extracting logo reference image SIFT features...')
logoImages=extractFeaturesBulk('./Data/logos/');
fprintf('done\n')

% FGCT Parameters
dthr        = 0.4;
alpha       = 1.2;
sigma       = 4;
threshold   = 0.08;

fprintf('Calculating correspondances for every test image for every logo...')
tic
% It can be used: parfor
for i=1:length(testImages)
    temp=zeros(1,length(logoImages));
    for j=1:length(logoImages)
        % Procedure to match features between image and logo
        % Matching is done using exhaustive search. Using LSH methods
        % the process time is reduced about 10 times.
        pairs = matchFeatures(testImages(i),logoImages(j),dthr);
        % Procedure that calculate the correspondance using FGCT
        temp(1,j) = FGCT(testImages(i),logoImages(j),pairs,alpha,sigma);
    end
    correspondance(i,:) = temp;
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
disp(['Average execution time: ' num2str(1000*d/(numel(correspondance))) 'ms per image'])
