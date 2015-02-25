%%%
% Parametrization file
%

% Data resources paths
dataPath = '../MSRAction3DSkeletonReal3D/';
addpath(genpath(dataPath));

% Third-party libraries' paths
hmmLibPath = '../Libs/HMMall/';
rmpath(genpath(hmmLibPath));

% My auxiliar libraries' paths
addpath('featuring/');
addpath('filtering/');
addpath('validating/');
addpath('normalizing/');
addpath('projection/');

%% Data structures, i.e. [actions x subjects x examples]
categories = [1:20];
subjects = [1:10];
examples = [1:3];
useConfidences = 0;

%% Filtering
movAvgLag = 4; % Moving average past frames to take into account ('lag')

%% Feature extraction
velOffset = 2; % Velocity computation respect to position in 'offset' frames ago

%% HMM (continuous) model parametrization

classifierName = 'TiedMixContinuousLeftrightHMM';

% General HMM parametrisation
params.numHidStates     = [5, 5];
params.selfTransProb    = 0.7;
params.maxIter          = 50;
% (and parameters of a tiedmix continuous HMM)
params.tiedMixParams.normParams     = [1 0]; % data normalisation/scaling
params.tiedMixParams.projVar        = 0.9; % data projection/dim.red
params.tiedMixParams.emInit         = 'rnd'; % Mix components initial centers (random or kmeans)
params.tiedMixParams.covType        = 'diag'; % Mixtures cov matrices form