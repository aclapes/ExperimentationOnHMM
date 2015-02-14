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
            A(end+1) = mean(mean(accs));
        end
    end
end

figure(1); hold on; title('Num hidden states');
M = cell(length(numHidStates),1);
m = zeros(length(numHidStates),1);
for i = 1:length(numHidStates)
    h = numHidStates(i);
    for j = 1:length(P)
        if h == P{j}.numHidStates
            M{i} = [M{i}, A(j)];
        end
    end
    m(i) = mean(M{i});
end
bar(m);
xlim([0.5 length(m)+0.5]); 
ylabel('Accuracy');
xlabel('Num hidden states');
set(gca, 'XTickLabel', mat2cell(numHidStates,1));
grid on; 
hold off;

figure(2); hold on; title('Num. mixtures');
M = cell(length(numMixtures),1);
m = zeros(length(numMixtures),1);
for i = 1:length(numMixtures)
    n = numMixtures(i);
    for j = 1:length(P)
        if n == P{j}.numMixtures(1)
            M{i} = [M{i}, A(j)];
        end
    end
    m(i) = mean(M{i});
end
bar(m);
xlim([0.5 length(m)+0.5]); 
ylabel('Accuracy');
xlabel('Num mixtures');
% set(gca, 'XTickLabel', mat2cell(numHidStates,1));
grid on; 
hold off;