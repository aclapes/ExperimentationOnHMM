% Test1.m
%
% Experiments on a discrete HMM: study of effect of parameters on number of
% quantization vectors and number of hidden states. The self transition 
% probability is kept fixed.
%
% A LOSOCV is used in order to measure the goodnes of the parameters, not
% the out-of-sample accuracy (an outer CV would be required).


addpath(genpath('../MSRAction3DSkeletonReal3D/'));
rmpath(genpath('../Libs/HMMall/'));

addpath('featextract/');

%% Parametrization

% Data structures, i.e. [actions x subjects x examples]

actions = [1:20];
subjects = [1:10];
examples = [1:3];
useConfidences = 0;

% Codebook

numClusters = [20, 40, 60, 80, 100];

% Feature extraction

offset = 2; % Velocity computation respect to position in 'offset' frames ago
% (normalisation)
neckIdx = 3; % Index of the joint used for position normalisation (e.g. neck)
% (smoothing)
lag = 4; % Moving average past frames to take into account ('lag')

% HMM model parametrization

selfTransProb = 0.6;
numHidStates = [3, 4, 5, 6, 7, 8, 10, 15, 20];
maxIters = 50;


%% Load data

info = []; % Action info
data = {}; % Action data

for i = 1:length(actions)
    a = actions(i);
    for j = 1:length(subjects)
        s = subjects(j);
        for k = 1:length(examples)
            e = examples(k);
            % Get the indexed skeleton
            skeleton = getskt3d('../MSRAction3DSkeletonReal3D/',a,s,e,useConfidences);
%             isempty(skeleton)
            if ~isempty(skeleton)
                % Append to the data cell
                info = [info; a,s,e];
                data{end+1,1} = skeleton;
            end
        end
    end
end


%% Filtering

% Smooth skeletons
for i = 1:length(data)
    seq = data{i};
    filtSktSeq = tsmovavg( [repmat(seq(1,:),lag,1); seq] ,'e', lag, 1);
    data{i} = filtSktSeq((1+lag):end, :);
end


%% Feature extraction

for i = 1:length(data)
    seq = data{i}; % Time as rows and features as cols
    
    % Get normalized positions (relative to neck joint)
    neckSeq = seq(:,(3*(neckIdx-1)+1):3*(neckIdx) );
    normSktSeq = seq - repmat(neckSeq,1,20);
    
    % Wrist-elbow-shoulder left and right angles
    leftElbowAngles = anglesBetweenJoints(normSktSeq, 1, 8, 10);
    rightElbowAngles = anglesBetweenJoints(normSktSeq, 2, 9, 11);
    leftShoulderAngles = anglesBetweenJoints(normSktSeq, 8, 1, 3);
    rightShoulderAngles = anglesBetweenJoints(normSktSeq, 9, 2, 3);
    
    % Get velocities
    velocities = velocitiesInJoints(normSktSeq, offset);
    
    % Build a joint representation (early feature fusion)
    data{i} = [normSktSeq, velocities, leftElbowAngles, rightElbowAngles, leftShoulderAngles, rightShoulderAngles];
end

subjects = unique(info(:,2));


%% Learning and results savings
% Using a Leave-One-Out Cross-Validation (LOOCV)

outsampleAccs = zeros( length(numClusters)*length(numHidStates), size(subjects,2) );

for i = 1:length(subjects)
    u = subjects(i);

    infoTr = info(info(:,2) ~= u,:);
    infoTe = info(info(:,2) == u,:);

    dataTr = data(info(:,2) ~= u,:);
    dataTe = data(info(:,2) == u,:);

    percTr = size(dataTr,1)/size(data,1) * 100;
    percTe = size(dataTe,1)/size(data,1) * 100;

    display(['Sbj ', num2str(i), '. Total data: ', num2str(size(data,1)), ', (', ...
        num2str(percTr), '% train, ', num2str(percTe), '% test).']);

    for j = 1:length(numClusters)
        [obsTr, obsTe] = preprocess(dataTr, dataTe, numClusters(j));
        for k = 1:length(numHidStates) 
            outsampleAccs((j-1) * length(numHidStates) + k, i) = discreteLeftrightHMMTest(obsTr, obsTe, ...
                infoTr, infoTe, numClusters(j), numHidStates(k), selfTransProb, maxIters);
        end
    end
end


%% Showing results and visualization stuff

outsampleAvgs = mean(outsampleAccs,2);
R = reshape(outsampleAvgs,length(numClusters),length(numHidStates));
close all; surf(R); 
hold on; 
title('Performance on num clust and hid states');
ylabel('numClusters');
xlabel('numHidStates'); 
set(gca,'YTickLabel', num2cell(numClusters));
set(gca,'XTickLabel', num2cell(numHidStates));
grid on;
holf off;

[maxCls, maxClsInds] = max(R);
[maxHid, maxHidIdx] = max(maxCls);

maxAccX = maxClsInds(:,maxHidIdx);
maxAccY = maxHidIdx;
bestAcc = R(maxAccX, maxAccY);
bestCls = numClusters(maxAccX);
bestHid = numHidStates(maxAccY);