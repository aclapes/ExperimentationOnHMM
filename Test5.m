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

s = 1;
stclass = (nfo(1,:) == s);
ndclass = (nfo(1,:) ~= s);
nfo(1, stclass) = 1;
nfo(1, ndclass) = 2;

M = {numMixtures, numMixtures};
C = allcomb(M{:});

numReplicas = 3;

for i = 1:size(C,1)
    for r = 1:numReplicas
        results = validateTiedMixLeftrightHMM(data, nfo, ...
            repmat([numHidStates],1,2), selfTransProb, C(i,:), ...
            normParam, projVar, ...
            emInit, covType, ...
            maxIter, verbose);

        save(sprintf('output/results/T5/S%d_%.2f-%d-%.2f-%s-%s-%d_%s.mat', ...
            s, selfTransProb, ...
            normParam(1,1), projVar, ...
            emInit, covType, r, ...
            datestr(now, 30)), 'results');
    end
end