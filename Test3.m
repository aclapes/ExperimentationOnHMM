% Test3.m - Test a continuous HMM

addpath(genpath('../MSRAction3DSkeletonReal3D/'));
rmpath(genpath('../Libs/HMMall/'));

addpath('featextract/');

%% Parametrization

% Data structures, i.e. [actions x subjects x examples]

actions = [1:20];
subjects = [1:10];
examples = [1:3];
useConfidences = 0;

% Feature extraction

offset = 2; % Velocity computation respect to position in 'offset' frames ago
% (normalisation)
neckIdx = 3; % Index of the joint used for position normalisation (e.g. neck)
% (smoothing)
lag = 4; % Moving average past frames to take into account ('lag')

% HMM (continuous) model parametrization

selfTransProb = 0.7;
numHidStates = 6;
maxIters = 50;

numMixtures = 1; % Number of mixtures
covType = 'full';

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
            skeleton = getskt3d('../MSRAction3DSkeletonReal3D/',a,s,e, useConfidences);
%             isempty(skeleton)
            if ~isempty(skeleton)
                % Append to the data cell
                info = [info, [a;s;e]];
                data{1,end+1} = skeleton;
            end
        end
    end
end


%% Filtering

% Smooth skeletons
for i = 1:length(data)
    seqt = data{i}'; % Seq transposed because tsmovavg later assumes rows
    dummy = repmat(seqt(1,:),lag,1);
    filtSktSeq = tsmovavg( [dummy; seqt],'e', lag, 1); % assumes row-wise
    data{i} = filtSktSeq((1+lag):end, :)';
end


%% Feature extraction

% Normalizing joint index
normJointIdx = 7; % pelvic joint
normJointInds = false(size(data{1},1),1); % logical indexing for norm joint rows indication  
normJointInds((3*(normJointIdx-1)+1):3*(normJointIdx)) = true;

for i = 1:length(data)
    seq = data{i}; % Time as rows and features as cols
    
    % Get normalized positions (relative to neck joint)
    normJointSeq = seq(normJointInds, :);
    normSktSeq = seq - repmat(normJointSeq,20,1);
    
    % Wrist-elbow-shoulder left and right angles
    leftElbowAngles = anglesBetweenJoints(normSktSeq, 1, 8, 10);
    rightElbowAngles = anglesBetweenJoints(normSktSeq, 2, 9, 11);
    leftShoulderAngles = anglesBetweenJoints(normSktSeq, 8, 1, 3);
    rightShoulderAngles = anglesBetweenJoints(normSktSeq, 9, 2, 3);
    
    % Get velocities
    velocities = velocitiesInJoints(normSktSeq, offset);
    
    % Build a joint representation (early feature fusion)
    data{i} = [normSktSeq(~normJointInds,:); velocities; ...
        leftElbowAngles; rightElbowAngles; leftShoulderAngles; rightShoulderAngles];
end

subjects = unique(info(2,:));


%% Learning and results savings
% Using a Leave-One-Out Cross-Validation (LOOCV)
rng(74);
outsampleAccs = zeros( 1, length(subjects) );

for i = 1:length(subjects)
    u = subjects(i);

    infoTr = info(:,info(2,:) ~= u);
    infoTe = info(:,info(2,:) == u);
    dataTr = data(info(2,:) ~= u);
    dataTe = data(info(2,:) == u);

    display(['Sbj ', num2str(i), '. Total data: ', num2str(length(data)), ', (', ...
        num2str(length(dataTr)/length(data)), '% train, ', num2str(length(dataTe)/length(data)), '% test).']);

    outsampleAccs(i) = continuousLeftrightHMMTest(dataTr, dataTe, ...
                infoTr, infoTe, numHidStates, selfTransProb, numMixtures, covType, maxIters);
end
