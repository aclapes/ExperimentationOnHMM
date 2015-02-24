%% Vis8.m

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

numActions = 5;

%% Test 0

normParam = [1 0];
projVar = 0.9;
emInit = 'rnd';
covType = 'full';

numHidStates = 5;
numMixturesR = [1 2 3 5];
numMixtures = [numMixturesR 10 20 30];

numReplicas = 3;

tmp = repmat({numMixtures}, 1, 2);
C = allcomb(tmp{:});

resultsSmry = cell(size(C,1), numReplicas); % 2-D cell structure of results

listings = cell(numActions,1);
for k = 1:numActions
    listings{k} = dir(['output/results/TCA0/', num2str(k) ,'/*.mat']);
end

Pmclasses = zeros(2,numActions);
PRmclasses = Pmclasses;
Cdics = zeros(2,numActions);

Pdics = zeros(1,numActions);
PRdics = Pdics;
CRdics = zeros(2,numActions);

for k = 1:numActions
    list = listings{k};
    
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
        S = resultsSmry{i,1}.outsampleAccs(actions,:);
        S(isnan(S)) = 0;

        Aone = S;
        Amax = S;
        Aavg = S / numReplicas;

        for r = 2:numReplicas
            s = resultsSmry{i,r}.outsampleAccs(actions,:);
            s(isnan(s)) = 0;

            Amax = max(Amax,s);
            Aavg = Aavg + (s / numReplicas);
        end

        Pone(:,i) = mean(Aone,2);
        Pmax(:,i) = mean(Amax,2);
        Pavg(:,i) = mean(Aavg,2);
    end

    [maxVal, maxIdx] = max(mean(Pavg));
    Pmclasses(:,k) = Pavg(:,maxIdx);
    Pdics(k) = mean(Pavg(:,maxIdx));
    Cdics(:,k) = C(maxIdx,:)';
    
    CRinds = (C(:,1) < 10) & (C(:,2) < 10);
    CR = C(CRinds,:);
    PRmax = Pavg(:,CRinds); % restricted
    [maxVal, maxIdx] = max(mean(PRmax));
    PRmclasses(:,k) = PRmax(:,maxIdx);
    PRdics(k) = mean(PRmax(:,maxIdx));
    CRdics(:,k) = CR(maxIdx,:)';
end