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

numHidStates = [3; 5; 7];
numMixtures = [5; 10; 15];

dichotomies = {'D01', 'D02', 'D03', 'D04'};
warning('off','all');

A = zeros(length(numHidStates) * length(numMixtures), 2, length(dichotomies));
for d = 1:length(dichotomies)
    A_d = [];
    P = {};
    dirlist = dir(['output/results/T2/', dichotomies{d}]);
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
    A(:,:,d) = A_d;
end
A = mean(A,3);

figure(1);
hold on;
title('Num of mixtures effect');
M = cell(length(numMixtures),1);
for i = 1:length(numMixtures)
    n = numMixtures(i);
    for j = 1:length(P)
        if n == P{j}.numMixtures(1)
            M{i} = [M{i}; A(j,:)];
        end
    end
    M{i} = mean(M{i});
end
bar(cell2mat(M));
legend('Class 1 (a class)','Class 2 (rest)');
xlim([0.5 length(m)+0.5]); 
set( gca, 'XTickLabel', num2cell(numMixtures) );
grid on; 
hold off;

figure(2); 
hold on;
title('Num of hidden states effect');
M = cell(length(numHidStates),1);
for i = 1:length(numHidStates)
    h = numHidStates(i);
    for j = 1:length(P)
        if h == P{j}.numHidStates
            M{i} = [M{i}; A(j,:)];
        end
    end
    M{i} = mean(M{i});
end
bar(cell2mat(M));
legend('Class 1 (a class)','Class 2 (rest)');
xlim([0.5 length(m)+0.5]); 
set( gca, 'XTickLabel', num2cell(numHidStates) );
grid on; 
hold off;