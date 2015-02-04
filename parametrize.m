%%%
% Parametrization file
%

%% Data structures, i.e. [actions x subjects x examples]
actions = [1:20];
subjects = [1:10];
examples = [1:3];
useConfidences = 0;

%% Filtering
movAvgLag = 4; % Moving average past frames to take into account ('lag')

%% Feature extraction
velOffset = 2; % Velocity computation respect to position in 'offset' frames ago

%% HMM (continuous) model parametrization
numHidStates = 6;
selfTransProb = 0.7;
maxIter = 10;
covType = 'full';