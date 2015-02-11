% Test3.m - Test a continuous HMM

rmpath(genpath('../Libs/HMMall/'));

addpath('featuring/');
addpath('filtering/');
addpath('validating/');
addpath('normalizing');

addpath(genpath('../MSRAction3DSkeletonReal3D/'));
addpath(genpath('output/'));

%% Parametrization

parametrize;

%% Load data

try
    load('data');
catch
    [data, nfo] = loadData('../MSRAction3DSkeletonReal3D/', ...
    actions, subjects, examples, useConfidences);

    % Filter noise
    data = movingAverageFilter(data, movAvgLag);
    % Extract features instead of RAW data
    data = extractKinematicFeatures(data, velOffset);
    
    save('data.mat', 'data', 'nfo');
end

%% Test 1
% The same no. mixtures for each class model.

numHidStates = [3 4 5];
numMixtures = [1 2 3 5 7 10 15 20]; % test different values

A = [];
P = {};

warning('off','all');
dirlist = dir('output/results/T1');
for i = 1:length(dirlist)
    name = dirlist(i).name;
    if ~isdir(name)
        load(name);
        if exist('results', 'var')
            P{end+1} = results.params;
            
            accs = results.outsampleAccs;
            accs(isnan(accs)) = 0;
            A(i) = mean(mean(accs));
        end
    end
end