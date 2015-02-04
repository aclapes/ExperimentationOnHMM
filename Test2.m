% Test2.m (complementary to Test1.m)
%
% Experiments on a discrete HMM: study of effect of self transition prob
% parameter. Num of quantization vectors and num of hidden states are kept
% fixed.
%
% A LOSOCV is used in order to measure the goodness of the parameter, not
% the out-of-sample accuracy (an outer CV would be required).

addpath(genpath('../MSRAction3DSkeletonReal3D/'));
rmpath(genpath('../Libs/HMMall/'));

addpath('featextract/');

%% Parametrization

actions = [1:20];
subjects = [1:10];
examples = [1:3];
useConfidences = 0;

numClusters = 80;

neckIdx = 3;

lag = 4; % Moving average past frames to take into account ('lag')
offset = 2; % Velocity computation respect to position in 'offset' frames ago

% Define the generating model

selfTransProb = [0.9:-0.1:0.1];
numHidStates = 6;
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
    seq = data{i};
    
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

%% Leave-One-Out Cross-Validation (LOOCV)

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

    [obsTr, obsTe] = preprocess(dataTr, dataTe, numClusters);
    for j = 1:length(selfTransProb)
        outsampleAccs(j, i) = discreteLeftrightHMMTest(obsTr, obsTe, ...
            infoTr, infoTe, numClusters, numHidStates, selfTransProb(j), maxIters);
    end
end