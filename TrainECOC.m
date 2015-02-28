% TestCA1.m - Test a class-aggregated continuous HMM

% Output paths
addpath(genpath('output/'));


%% Parametrization

loadPathsAndGlobalParameters; % global parametrization

% Data-related parameters
numCategoriesInExpt = 5;

rng(42);
perm = randperm(length(categories));
categoriesInExpt = perm(1:numCategoriesInExpt);
inds(categoriesInExpt) = [1:length(categoriesInExpt)]; % used further

% Meta-classifier (ECOC) parameters

scheme = 'onevsall';
D = codingECOC(scheme, length(categoriesInExpt)); % ECOC dichotomies' coding matrix

% Validation parameters

tiedMixParams.numMixtures = [1 2 3 5 10 20];
numValidationFolds = 3;
verbose = 0;


%% Load data

if exist('data', 'file')
    load('data');
else
    [data, nfo] = loadData(dataPath, categoriesInExpt, subjects, examples, useConfidences);
    % % Filter noise
    % data = movingAverageFilter(data, movAvgLag); % perhaps no need
    % Extract features instead of RAW data
    data = extractKinematicFeatures(data, velOffset);
    save('data.mat', 'data', 'nfo');
end


%% Train and test

% no. mixtures is to be validated, thus generate the combinations
aux = repmat({tiedMixParams.numMixtures}, 1, 2);
C = allcomb(aux{:}); % if using ECOC, it is numMixtures^2

% Output results will be kept in a directory named as this script
outputDir = ['output/results/', mfilename]; % mfilename returns the script name
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
                
warning('off','all');

% For each test fold, a whole ECOC is build and internal validation of
% parameters is performed

for t = T     
    dataTr           = data(nfo.subjects ~= t);
    nfoTr.categories = nfo.categories(nfo.subjects ~= t);
    nfoTr.subjects   = nfo.subjects(nfo.subjects ~= t);
    
    dicValidation = cell( 1, size(D,2) ); 

    tmpDir = sprintf('%s/F%d-tmp', outputDir, t);
    if ~exist(tmpDir, 'dir')
        mkdir(tmpDir);
    end
    disp(sprintf('In fold %d', t));

    for d = 1:size(D,2)
        dicIndsTr = D(inds(nfoTr.categories),d)'; % double LUT(-ing)
        
        dicDataTr           = [dataTr(dicIndsTr == 1), dataTr(dicIndsTr == 2)];
        dicNFOTr.categories = [dicIndsTr(dicIndsTr == 1),  dicIndsTr(dicIndsTr == 2)];
        dicNFOTr.subjects   = [nfoTr.subjects(dicIndsTr == 1), nfoTr.subjects(dicIndsTr == 2)];
        
        % train and keep models, that are {combinations x internal folds x
        % metaclasses} structures
        dicValModels = cell(size(C,1), numValidationFolds); % i.e. all the models in the dichotomy
        dicValAccuracies = zeros(size(C,1), numValidationFolds);
        
        % (inner) k-fold cross validation 
        % (stratified by actions and subjects altogether)
        CVO = cvpartition(dicNFOTr.categories .* dicNFOTr.subjects, 'KFold', numValidationFolds);
        
        disp(sprintf('Processing dichotomy %d/%d ..', d, size(D,2)));
        
        TSTART = tic;
        for j = 1:CVO.NumTestSets
            
            dicDataTrTr            = dicDataTr(CVO.training(j));
            dicNFOTrTr.categories  = dicNFOTr.categories(CVO.training(j));
            dicNFOTrTr.subjects    = dicNFOTr.subjects(CVO.training(j));
            
            dicDataVal            = dicDataTr(CVO.test(j));
            dicNFOVal.categories  = dicNFOTr.categories(CVO.test(j));
            dicNFOVal.subjects    = dicNFOTr.subjects(CVO.test(j));
                                  
            % train the combinations in the partition
            for i = 1:size(C,1)                
                params.tiedMixParams.numMixtures = C(i,:); % only validating this param
                [dicProcDataTrTr, dicProcDataVal] = preprocessData(dicDataTrTr, params, dicDataVal);
                
                % train model and validate its performance
                addpath(genpath(hmmLibPath));
                lambdas = trainTiedMixLeftrightHMM(dicProcDataTrTr, dicNFOTrTr, params, verbose); 
                [preds.metacategories, preds.loglikes] = testTiedMixLeftrightHMM(lambdas, dicProcDataVal);
                rmpath(genpath(hmmLibPath)); % interfieres with MATLAB functions (e.g. princomp)
                
                dicValModels{i,j}.params    = params;
                dicValModels{i,j}.lambdas   = lambdas;
                dicValModels{i,j}.preds     = preds;
                dicValAccuracies(i,j) = accuracy(dicNFOVal.categories, preds.metacategories);
            end
        end
        dicElapsedTime = toc(TSTART); 
        disp(sprintf('It took %.2f s.', dicElapsedTime));
        
        % Save dichotomy's results in a file
        dicValidation.CVO           = CVO;
        dicValidation.categories    = categoriesInExpt;
        dicValidation.dic           = D(:,d);
        dicValidation.valModels     = dicValModels;
        dicValidation.valAccuracies = dicValAccuracies;
        dicValidation.elapsedTime   = dicElapsedTime;
        
        filepath = sprintf('%s/d%d.mat', tmpDir, d);
        save(filepath, 'dicValidation');
        
        % Save in the general structure
        dicValModels{d} = dicValidation;
    end
     
    % Save the test fold's results in a file
    trainFoldOutput.classifier                          = classifierName;
    trainFoldOutput.scheme                              = scheme;
    trainFoldOutput.categories                          = categoriesInExpt;
    trainFoldOutput.D                                   = D;
    trainFoldOutput.params                              = params;
    trainFoldOutput.valParams.tiedMixParams.numMixtures = tiedMixParams.numMixtures;
    trainFoldOutput.numValidationFolds                  = numValidationFolds;
    trainFoldOutput.dicValModels                        = dicValModels;
    
    filepath = sprintf('%s/F%d.mat', outputDir, t);
    save(filepath, 'trainFoldOutput');
    
    % Remove temporary files
    delete([tmpDir, '/*.mat']);
end

% for t = T   
%     
%     dataTr           = data(nfo.subjects ~= t);
%     nfoTr.categories = nfo.categories(nfo.subjects ~= t);
%     nfoTr.subjects   = nfo.subjects(nfo.subjects ~= t);
% 
%     dataTe           = data(nfo.subjects == t);
%     nfoTr.categories = nfo.categories(nfo.subjects == t);
%     nfoTr.subjects   = nfo.subjects(nfo.subjects == t);
%     
%     
%     
% end

% for t = 1:length(actions)    
%     dataTe = data(nfo(2,:) == t);
%     nfoTe = nfo(:, nfo(2,:) == t);
%     
%     for d = 1:length(actions)
%         dicIndsTe = D(inds(nfoTe(1,:)),d)'; % again
%         dicDataTe = [dataTe(dicIndsTe == 1), dataTe(dicIndsTe == 2)];
%         dicNfoTe = [nfoTe(:,dicIndsTe == 1), nfoTe(:,dicIndsTe == 2)];
%         
%     end
% end

%         dataTrTr = dataTr(

%         for i = 1:size(C)
%             TSTART = tic;
%             results = validateTiedMixLeftrightHMM(dicdata, [dicinds(dicinds > 0); nfo(2:end, dicinds > 0)], ...
%                 repmat([numHidStates],1,2), selfTransProb, C(i,:), ...
%                 normParam, projVar, ...
%                 emInit, covType, ...
%                 maxIter, verbose);
%             results.classes = actions';
%             results.metaclasses = D(:,d);
%             results.time = toc(TSTART);
% 
%             filename = sprintf(['%d-%.2f-', repmat(['%d-'], 1, 2),'%d-%.2f-%s-%s_%s.mat'], ...
%                 numHidStates, selfTransProb, C(i,:), ...
%                 normParam(1,1), projVar, ...
%                 emInit, covType, ...
%                 datestr(now, 30));
% 
%             dicOutputDir = sprintf(['%sD', repmat('%d',1,length(D(:,d))), '/'], outputDir, D(:,d));
%             if ~exist(dicOutputDir, 'dir')
%                 mkdir(dicOutputDir);
%             end
%             save([dicOutputDir, filename], 'results');
% 
%             fprintf('%s took %.3f s.\n', filename, results.time);
%         end