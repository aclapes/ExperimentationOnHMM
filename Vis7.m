%% Vis6.m
% Illustrate the results from script Test6.m, with the results
% in outputs/results/T6.
%
% We can select a subset of actions by modifying the variable actions,
% e.g. to select the subjset of actions 1 to 4, define: actions = [1:4]
% somewhere.

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

%% Test 0

normParam = [1 0];
projVar = 0.9;
emInit = 'rnd';
covType = 'full';

numHidStates = 5;
numMixtures = [1 2 3 5];

numReplicas = 4;

tmp = repmat({numMixtures}, 1, length(actions));
C = allcomb(tmp{:});

resultsSmry = cell(size(C,1), numReplicas); % 2-D cell structure of results

list = dir('output/results/ST6/*.mat');

% Load results mats
for i = 1:size(C,1)
    for r = 1:numReplicas
        idx = (i-1) * numReplicas + r;

        load(list(idx).name, 'results');
        resultsSmry{i,r} = results;
    end
end

actions = unique(resultsSmry{i,r}.nfo(1,:));

% Process the results (for the specified subset of actions)

Pone = zeros(length(actions),size(C,1));
Pavg = zeros(length(actions),size(C,1));
Pmax = zeros(length(actions),size(C,1));

for i = 1:size(C,1)
    S = resultsSmry{i,1}.outsampleAccs;
    S(isnan(S)) = 0;
    
    Aone = S;
    Amax = S;
    Aavg = S / numReplicas;
    
    for r = 2:numReplicas
        s = resultsSmry{i,r}.outsampleAccs();
        s(isnan(s)) = 0;
        
        Amax = max(Amax,s);
        Aavg = Aavg + (s / numReplicas);
    end
    
    Pone(:,i) = mean(Aone,2);
    Pmax(:,i) = mean(Amax,2);
    Pavg(:,i) = mean(Aavg,2);
end

