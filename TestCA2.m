% TestCA1.m - Test a class-aggregated continuous HMM

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

numMixtures = [1 2 3 5 10 20];

normParam = [1 0];
projVar = 0.9;
emInit = 'rnd';
covType = 'diag';

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

%% Test CA1
% A different number of mixtures per class

tmp = repmat({numMixtures}, 1, 2);
C = allcomb(tmp{:});

warning('off','all');

if ~exist(['output/results/', mfilename])
    mkdir(['output/results/', mfilename]);
end

D = -eye(length(actions)) + 2;

for d = vargin
    metalbls = zeros(size(nfo(1,:)));
    metalbls(nfo(1,:) == actions(d)) = 1;
    metalbls(nfo(1,:) ~= actions(d)) = 2;
    
    for i = 1:size(C)
        TSTART = tic;
        [results, models] = validateTiedMixLeftrightHMM(data, [metalbls; nfo(2:end,:)], ...
            repmat([numHidStates],1,2), selfTransProb, C(i,:), ...
            normParam, projVar, ...
            emInit, covType, ...
            maxIter, verbose);
        results.time = toc(TSTART);

        filename = sprintf(['%d-%.2f-', repmat(['%d-'], 1, 2),'%d-%.2f-%s-%s-', repmat(['%d'], 1, size(D,2)), '_%s.mat'], ...
            numHidStates, selfTransProb, C(i,:), ...
            normParam(1,1), projVar, ...
            emInit, covType, D(:,d), ...
            datestr(now, 30));

        save(['output/results/', mfilename, '/results_', filename], 'results');
        save(['output/models/', mfilename, '/models_', filename], 'models');

        fprintf('%s took %.3f s.\n', filename, results.time);
    end
end