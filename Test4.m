% Test3.m - Test a continuous HMM

rmpath(genpath('../Libs/HMMall/'));

addpath('featextract/');
addpath('filtering/');

addpath(genpath('../MSRAction3DSkeletonReal3D/'));
addpath(genpath('output/'));

%% Parametrization

parametrize;
% L = actions; % multi-class

% auxL = [ tril(ones(length(actions)/2))+triu(2*ones(length(actions)/2),1), 2*ones(length(actions)/2) ];
% rng(74);
% L = flipud( auxL(:,randperm(size(auxL,2))) );

L = [1 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2];

%% Load data

load('data');
if ~exist('data','var')
    [data, nfo] = loadData('../MSRAction3DSkeletonReal3D/', ...
    actions, subjects, examples, useConfidences);

    % Filter noise
    data = movingAverageFilter(data, 4);
    % Extract features instead of RAW data
    data = extractKinematicFeatures(data, 2);
    
    save('data.mat', 'data', 'nfo');
end


%% Learning and results savings
% Using a Leave-One-Out Cross-Validation (LOOCV)

subjects = unique(nfo(2,:));

for l = 1:size(L,1)     
    rng(74);

    preds = zeros(1,size(nfo,2));
    likes = cell(1,size(nfo,2));
    paths = cell(1,size(nfo,2));
    
    lut = L(l,:); % look-up table
    dicnfo = [lut(nfo(1,:)); nfo(2:end,:)];
    
    classes = unique(dicnfo(1,:));
    
    instances = histc(lut(:),classes);
    numMixtures = floor(1.5*(log2(instances)+1)); % min( floor(instances/2.0)+1, repmat([5], length(instances), 1));
%     [maxVal, maxIdx] = max(histc(dicnfo(:),classes));
%     numMixtures = 2 * (floor( log(sum(lut==classes(maxIdx))) + 1));
    
    outsampleAccs = nan( length(classes), length(subjects) );

    for i = 1:length(subjects)
        u = subjects(i);

        indicesTr = dicnfo(2,:) ~= u; % leave i-th subject out
        indicesTe = dicnfo(2,:) == u;
        
        dicinfoTr = dicnfo(:,indicesTr);
        dicinfoTe = dicnfo(:,indicesTe);
        dataTr = data(indicesTr);
        dataTe = data(indicesTe);

        display(['Sbj ', num2str(i), '. Total data: ', num2str(length(data)), ', (', ...
            num2str(length(dataTr)/length(data)), '% train, ', num2str(length(dataTe)/length(data)), '% test).']);

        [predsTe, likesTe, pathsTe] = continuousLeftrightHMMTest(dataTr, dataTe, ...
                    dicinfoTr, dicinfoTe, numHidStates, selfTransProb, numMixtures, covType, maxIters);

        preds(indicesTe) = predsTe;
        likes(indicesTe) = {likesTe};
        paths(indicesTe) = pathsTe;
        
        [~, A, C] = accuracy(dicinfoTe(1,:), predsTe);
        outsampleAccs(C,i) = A;
    end
    
    save([num2str(lut(:))', '.mat'], 'lut', 'preds', 'likes', 'paths', ...
        'outsampleAccs', 'numHidStates', 'dicnfo', 'numMixtures');

end
