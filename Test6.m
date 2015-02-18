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

rng(42);
inds = randperm(20);
actions = inds(1:5);
rng('shuffle');

numMixtures = [1 3 9 27];

normParam = [1 0];
projVar = 0.9;
emInit = 'rnd';
covType = 'full';

numReplicas = 3;
verbose = 0;

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

%% Test 6
% A different number of mixtures per class

tmp = repmat({numMixtures}, 1, length(actions));
C = allcomb(tmp{:});

warning('off','all');

for i = 1:size(C)
    TSTART = tic;
    results = validateTiedMixLeftrightHMM(data, nfo, ...
        repmat([numHidStates],1,length(actions)), selfTransProb, C(i,:), ...
        normParam, projVar, ...
        emInit, covType, ...
        maxIter, verbose);
    results.time = toc(TSTART);

    filename = sprintf(['%d-%.2f-', repmat(['%d-'], 1, length(actions)),'%d-%.2f-%s-%s_%s.mat'], ...
        numHidStates, selfTransProb, C(i,:), ...
        normParam(1,1), projVar, ...
        emInit, covType, ...
        datestr(now, 30));

    save(['output/results/ST6/', filename], 'results');
    fprintf('%s took %.3f s.\n', filename, results.time);
end