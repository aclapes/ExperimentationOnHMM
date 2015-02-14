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

A = [];
P = {};

warning('off','all');
dirlist = dir('output/results/T0/');
for i = 1:length(dirlist)
    name = dirlist(i).name;
    if ~isdir(name) && ~strcmp(name,'README.txt')
        load(name);
        if exist('results', 'var')
            P{end+1} = results.params;
            
            accs = results.outsampleAccs;
            accs(isnan(accs)) = 0;
            A = [A; mean(mean(accs))];
        end
    end
end

%% Per-parameter visualisation

colors = {'r','g','b','m','y','c','w','k'};

figure(1); title('Normalisation and scaling effect');
M = cell(size(normParams,1),1);
L = {};
m = zeros(size(normParams,1),1);
for i = 1:size(normParams,1)
    n = normParams(i,1);
    p = normParams(i,2);
    l = {};
    for j = 1:length(P)
        if n == P{j}.normParams(1,1) && p == P{j}.normParams(1,2)
            M{i} = [M{i}, A(j)];
            l{end+1} = P{j};
        end
    end
    L{end+1} = l;
    m(i) = mean(M{i});
    hold on;
    bar(i,m(i),colors{n+1});
end
legend('None','Minmax', 'Std (0.1)', 'Std', 'Std (10)', 'Unit (1-norm)', 'Unit (2-norm)');
xlim([0.5 length(m)+0.5]); 
ylabel('Accuracy');
grid on; 
hold off;

MM = cell2mat(M);
[maxRow, maxRowInds] = max(MM);
[maxElm, maxColIdx] = max(maxRow);
maxRowIdx = maxRowInds(maxColIdx);
L{maxRowIdx}{maxColIdx} 
MM(maxRowIdx,maxColIdx)

figure(2); title('PCA decorrelation and dim reduction effect');
M = cell(size(projVars,1),1);
m = zeros(size(projVars,1),1);
for i = 1:size(projVars,1)
    v = projVars(i);
    for j = 1:length(P)
        if v == P{j}.projVar
            M{i} = [M{i}, A(j)];
        end
    end
    m(i) = mean(M{i});
    hold on;
    bar(i,m(i),colors{i});
end
legend('None','0.75', '0.9');
xlim([0.5 length(m)+0.5]); 
ylabel('Accuracy');
grid on; 
hold off;

figure(3);
subplot(1,2,1); hold on; title('EM initialisation effect (all scalings)');
M = cell(size(emInits,1),1);
m = zeros(size(emInits,1),1);
for i = 1:size(emInits,1)
    init = emInits(i);
    for j = 1:length(P)
        if strcmp(init,P{j}.emInit)
            M{i} = [M{i}, A(j)];
        end
    end
    m(i) = mean(M{i});
    hold on;
    bar(i,m(i),colors{i});
end
legend('rnd','kmeans');
xlim([0.5 length(m)+0.5]); 
ylim([0 1]);
ylabel('Accuracy');
grid on; 
hold off;

subplot(1,2,2); hold on; title('EM initialisation effect (minmax)');
M = cell(size(emInits,1),1);
m = zeros(size(emInits,1),1);
for i = 1:size(emInits,1)
    init = emInits(i);
    for j = 1:length(P)
        if P{j}.normParams(1,1) == 1 && strcmp(init,P{j}.emInit)
            M{i} = [M{i}, A(j)];
        end
    end
    m(i) = mean(M{i});
    hold on;
    bar(i,m(i),colors{i});
end
legend('rnd','kmeans');
xlim([0.5 length(m)+0.5]); 
ylim([0 1]);
ylabel('Accuracy');
grid on; 
hold off;


figure(4);
subplot(1,2,1); hold on; title('Covariance matrix shape effect (all scalings)');
M = cell(size(covTypes,1),1);
m = zeros(size(covTypes,1),1);
for i = 1:size(covTypes,1)
    covType = covTypes(i);
    for j = 1:length(P)
        if strcmp(covType,P{j}.covType)
            M{i} = [M{i}, A(j)];
        end
    end
    m(i) = mean(M{i});
    hold on;
    bar(i,m(i),colors{i});
end
legend('diag','full');
xlim([0.5 length(m)+0.5]); 
ylim([0 1]);
ylabel('Accuracy');
grid on; 
hold off;

subplot(1,2,2); hold on; title('Covariance matrix shape effect (minmax)');
M = cell(size(covTypes,1),1);
m = zeros(size(covTypes,1),1);
for i = 1:size(covTypes,1)
    covType = covTypes(i);
    for j = 1:length(P)
        if P{j}.normParams(1,1) == 1 && strcmp(covType,P{j}.covType)
            M{i} = [M{i}, A(j)];
        end
    end
    m(i) = mean(M{i});
    hold on;
    bar(i,m(i),colors{i});
end
legend('diag','full');
xlim([0.5 length(m)+0.5]); 
ylim([0 1]);
ylabel('Accuracy');
grid on; 
hold off;