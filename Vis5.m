%% Vis5.m
% Illustrate the results from script Test5.m, with the results
% in outputs/results/T5.

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

load('data');
actions = [1:4]; % Specify a subset of actions

%% Test 0
% The same no. mixtures for each class model.

normParam = [1 0];
projVar = 0.9;
emInit = 'rnd';
covType = 'full';

numHidStates = 5;
numMixtures = [1 3 9 27];

M = {numMixtures, numMixtures};
C = allcomb(M{:});

ocdics = [1:4];
numReplicas = 3;

resultsSmry = cell(length(ocdics),length(C),numReplicas); % 3-D cell structure of results

% Load results mats
for i = 1:length(ocdics)
    for j = 1:size(C,1)
        for r = 1:numReplicas
            list = dir([sprintf('output/results/T5/S%db_%d-%.2f-%d-%d-%d-%.2f-%s-%s-%d', ...
                ocdics(i), numHidStates, selfTransProb, C(j,:), ...
                normParam(1,1), projVar, ...
                emInit, covType, r), '*.mat']);
            if ~isempty(list)
                load(list(1).name, 'results');
                resultsSmry{i,j,r} = results;
            end
        end
    end
end

Rone = zeros(length(ocdics),length(C)); % 3-D mat structure of results
Rrest = zeros(length(ocdics),length(C));
for i = 1:length(ocdics)
    for j = 1:size(C,1)
        for r = 1:numReplicas
            A = resultsSmry{i,j,r}.outsampleAccs;
            A(isnan(A)) = 0;
            
            m = mean(A,2) / numReplicas;
            Rone(i,j) = Rone(i,j) + m(1,:);
            Rrest(i,j) = Rrest(i,j) + m(2,:);
        end
    end
end

R5one = reshape(mean(Rone),length(numMixtures), length(numMixtures));
R5rest = reshape(mean(Rrest),length(numMixtures), length(numMixtures));