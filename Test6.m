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

numHidStates = 5;
numMixtures = [1 3 9 27];

warning('off','all');


numReplicas = 3;

for i = 1:length(numMixtures)
    for r = 1:numReplicas
        results = validateTiedMixLeftrightHMM(data, nfo, ...
            repmat([numHidStates],1,length(actions)), selfTransProb, repmat([numMixtures(i)], 1, length(actions)), ...
            normParam, projVar, ...
            emInit, covType, ...
            maxIter, verbose);

        save(sprintf('output/results/T6/%d-%.2f-%d-%d-%.2f-%s-%s-%d_%s.mat', ...
            numHidStates, selfTransProb, numMixtures(i), ...
            normParam(1,1), projVar, ...
            emInit, covType, r, ...
            datestr(now, 30)), 'results');
    end
end