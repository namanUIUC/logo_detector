% BLOB DETECTOR 
%% NAMAN SHUKLA

clear all
clc

% Path to the folder and subfolder
root_path = 'data/';
subject_name = 'xyz.jpg';                

% Hyper parameters (if any)
Sigma = 2;                                      % User defined
ScaleStep = 12;                                 % User defined
threshold = 0.2;                              % Typically 0.00001 - 0.1
method = 'Down';                               % vals: 'Up', 'Down'
time = true;                                   % vals: true, false
implementation = 0;                            % vals: BruteForce=0, DoG=1 [NOT SUPPORT]

%% IMAGE INPUT
full_path = sprintf('%s%s', root_path, subject_name);
image = imread(full_path);

%% BLOB DETECTION
BlobDetection(image, Sigma, ScaleStep, threshold, method, implementation, time)

