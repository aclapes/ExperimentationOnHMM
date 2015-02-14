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

%% Test 2

warning('off','all');

A_d = [];
P = {};
dirlist = dir(['output/results/T4/']);
for i = 1:length(dirlist)
    name = dirlist(i).name;
    if ~isdir(name)
        load(name);
        if exist('results', 'var')
            P{end+1} = results.params;

            accs = results.outsampleAccs;
            accs(isnan(accs)) = 0;
            A_d = [A_d; mean(accs,2)'];
        end
    end
end
A = mean(A_d)