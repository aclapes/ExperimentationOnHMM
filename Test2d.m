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

numHidStates = [3 5 7];
numMixtures = [5 10 15];

warning('off','all');

d = 4;
stclass = (nfo(1,:) == d);
ndclass = (nfo(1,:) ~= d);
nfo(1, stclass) = 1;
nfo(1, ndclass) = 2;

for j = 1:length(numHidStates)
    for i = 1:length(numMixtures)
        results = validateTiedMixLeftrightHMM(data, nfo, ...
            numHidStates(j), selfTransProb, repmat(numMixtures(i), length(actions), 1), ...
            normParam, projVar, ...
            emInit, covType, ...
            maxIter, verbose);

        save(sprintf('output/results/T2/T2_%d-%.2f-%d-%d-%.2f-%.2f-%s-%s_%s.mat', ...
            numHidStates(j), selfTransProb, numMixtures(i), ...
            normParam(1,1), projVar, ...
            emInit, covType, ...
            datestr(now, 30)), 'results');
    end
end