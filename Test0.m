% Test3.m - Test a continuous HMM

rmpath(genpath('../Libs/HMMall/'));

addpath('featuring/');
addpath('filtering/');
addpath('validating/');
addpath('normalizing/');
addpath('projection/');

addpath(genpath('../MSRAction3DSkeletonReal3D/'));
addpath(genpath('output/'));

%% Parametrization

parametrize;

%% Load data

if exist('data', 'file')
    load('data');
else
    [data, nfo] = loadData('../MSRAction3DSkeletonReal3D/', ...
    actions, subjects, examples, useConfidences);

    % Filter noise
%     data = movingAverageFilter(data, movAvgLag);
    % Extract features instead of RAW data
    data = extractKinematicFeatures(data, velOffset);
    
    save('data.mat', 'data', 'nfo');
end

%% Test 0
% The same no. mixtures for each class model.

numMixtures = [1]; %3 5 7 9 11 13 15]; % test different values

warning('off','all');
for i = 1:size(numMixtures,2)
    results = validateTiedMixLeftrightHMM(data, nfo, numHidStates, selfTransProb, ...
        repmat(numMixtures(i), length(actions), 1), covType, maxIter, ...
        standardization, scale, reduction, verbose);
    
    save(sprintf('output/results/T0_%d-%.2f-%d-%d-%.2f_%s.mat', ...
        numHidStates, selfTransProb, numMixtures(i), ...
        standardization, reduction, datestr(now, 30)), 'results');
end