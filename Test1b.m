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

normParam = [1 0];
projVar = 0.9;
emInit = 'rnd';
covType = 'full';

numMixtures = [1 2 3 5 7 10 15 20];

warning('off','all');
for i = 3:4
    results = validateTiedMixLeftrightHMM(data, nfo, ...
        numHidStates, selfTransProb, repmat(numMixtures(i), length(actions), 1), ...
        normParam, projVar, ...
        emInit, covType, ...
        maxIter, verbose);
    
    save(sprintf('output/results/T1/T1_%d-%.2f-%d-%d-%.2f-%.2f-%s-%s_%s.mat', ...
        numHidStates, selfTransProb, numMixtures(i), ...
        normParam(1,1), projVar, ...
        emInit, covType, ...
        datestr(now, 30)), 'results');
end