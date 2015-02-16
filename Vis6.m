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

numReplicas = 3;

resultsSmry = cell(length(numMixtures),numReplicas); % 2-D cell structure of results

% Load results mats
for i = 1:length(numMixtures)
    for r = 1:numReplicas
        list = dir([sprintf('output/results/T6/%d-%.2f-%d-%d-%.2f-%s-%s-%d', ...
            numHidStates, selfTransProb, numMixtures(i), ...
            normParam(1,1), projVar, ...
            emInit, covType, r), '*.mat']);
        if ~isempty(list)
            load(list(1).name, 'results');
            resultsSmry{i,r} = results;
        end
    end
end

% Process the results (for the specified subset of actions)

R = zeros(length(actions),length(numMixtures)); % actions x mixtures

for i = 1:length(numMixtures)
    for r = 1:numReplicas
        A = resultsSmry{i,r}.outsampleAccs(actions, :);
        A(isnan(A)) = 0; % if nan, any example of examples' subset (action,subject) was predicted (bc of errors)
        
        R(:,i) = R(:,i) + ( mean(A(actions,:),2) / numReplicas );
    end
end

R6 = mean(R);

% close all;
% bar3([mean(R);R]);
% hold on;
% title('Performance of one-class HMM (function of no. mixtures)');
% xlabel('Num mixtures');
% ylabel('ActionID');
% zlabel('Accuracy');
% legend(cellstr(num2str(numMixtures', 'M=%d')));
% xlim([.5 length(numMixtures)+.5]);
% ylim([.5 length(actions)+.5+1]);
% grid on;
% hold off;