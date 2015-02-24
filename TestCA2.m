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
perm = randperm(20);
actions = perm(1:5);
inds(actions) = [1:length(actions)]; % used further
rng('shuffle');

numMixtures = [1 2 3 5 10 20];

normParam = [1 0];
projVar = 0.9;
emInit = 'rnd';
covType = 'diag';

numReplicas = 3;
verbose = 0;

schemeECOC = 'onevsone';

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

D = codingECOC(schemeECOC, length(actions));
outputDir = ['output/results/', mfilename, '/', schemeECOC, '/'];

for d = 1:length(actions)
    
    dicinds = D(inds(nfo(1,:)),d)';
    dicdata = [data(dicinds == 1), data(dicinds == 2)];
    
    for i = 1:size(C)
        TSTART = tic;
        results = validateTiedMixLeftrightHMM(dicdata, [dicinds(dicinds > 0); nfo(2:end, dicinds > 0)], ...
            repmat([numHidStates],1,2), selfTransProb, C(i,:), ...
            normParam, projVar, ...
            emInit, covType, ...
            maxIter, verbose);
        results.classes = actions';
        results.metaclasses = D(:,d);
        results.time = toc(TSTART);

        filename = sprintf(['%d-%.2f-', repmat(['%d-'], 1, 2),'%d-%.2f-%s-%s_%s.mat'], ...
            numHidStates, selfTransProb, C(i,:), ...
            normParam(1,1), projVar, ...
            emInit, covType, ...
            datestr(now, 30));

        dicOutputDir = sprintf(['%sD', repmat('%d',1,length(D(:,d))), '/'], outputDir, D(:,d));
        if ~exist(dicOutputDir, 'dir')
            mkdir(dicOutputDir);
        end
        save([dicOutputDir, filename], 'results');

        fprintf('%s took %.3f s.\n', filename, results.time);
    end
end