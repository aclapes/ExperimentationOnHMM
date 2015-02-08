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

normParams = [
    0 0; 
    1 0; 
    2 0.1; 2 1; 2 10; 
    3 1; 3 2];
projVars = [0; 0.75; 0.9];
emInits = {'rnd';'kmeans'};
covTypes = {'diag';'full'};

indsParams = {[1:size(normParams,1)], 
    [1:length(projVars)], 
    [1:length(emInits)], 
    [1:length(covTypes)]};

combsInds = allcomb(indsParams{:});

warning('off','all');
for i = 1:size(combsInds,1)
    results = validateTiedMixLeftrightHMM(data, nfo, ...
        numHidStates, selfTransProb, repmat(2, length(actions), 1), ...
        normParams(combsInds(i,1),:), projVars(combsInds(i,2)), ...
        emInits{combsInds(i,3)}, covTypes{combsInds(i,4)}, ...
        maxIter, verbose);
    
    save(sprintf('output/results/T0_%d-%.2f-%d-%d-%.2f-%.2f-%s-%s_%s.mat', ...
        numHidStates, selfTransProb, numMixtures(i), ...
        normParams(combsInds(i,1),:), projVars(combsInds(i,2)), ...
        emInits{combsInds(i,3)}, covTypes{combsInds(i,4)}, ...
        datestr(now, 30)), 'results');
end