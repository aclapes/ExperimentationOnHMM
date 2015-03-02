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

scheme = 'MLE';

% Validation parameters

tiedMixParams.numMixtures = [1 2 3 5];
numValidationFolds = 3;
verbose = 0;

% Test parameters

numFinalModelReplicas = 3;


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
aux = repmat({tiedMixParams.numMixtures}, 1, length(categoriesInExpt));
C = allcomb(aux{:});

% Output results will be kept in a directory named as this script
outputDir = ['output/results/', mfilename]; % mfilename returns the script name
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
                
warning('off','all');


%% DICHOTOMIES' VALIDATION MODELS TRAINING

for i = 1:length(subjects)  
    t = subjects(i);     

    filepath = sprintf('%s/F%d.mat', outputDir, t);
    if exist(filepath, 'file')
        continue;
    end
    
    dataTr           = data(nfo.subjects ~= t);
    nfoTr.categories = nfo.categories(nfo.subjects ~= t);
    nfoTr.subjects   = nfo.subjects(nfo.subjects ~= t);
    
    tmpDir = sprintf('%s/F%d-tmp', outputDir, t);
    if ~exist(tmpDir, 'dir')
        mkdir(tmpDir);
    end
    disp(sprintf('In fold %d', t));

    CVO = cvpartition(nfoTr.categories .* nfoTr.subjects, 'KFold', numValidationFolds);
    
    valModels = cell(size(C,1), CVO.NumTestSets);
    valAccuracies = zeros(size(C,1), CVO.NumTestSets);

    TSTART = tic;
    for j = 1:CVO.NumTestSets

        dataTrTr            = dataTr(CVO.training(j));
        nfoTrTr.categories  = nfoTr.categories(CVO.training(j));
        nfoTrTr.subjects    = nfoTr.subjects(CVO.training(j));

        dataVal            = dataTr(CVO.test(j));
        nfoVal.categories  = nfoTr.categories(CVO.test(j));
        nfoVal.subjects    = nfoTr.subjects(CVO.test(j));

        for i = 1:size(C,1) 
            TSTART2 = tic;
            params.tiedMixParams.numMixtures = C(i,:); % only validating this param
            [dicProcDataTrTr, ~, dicProcDataVal] = preprocessData(dataTrTr, params.preprocParams, dataVal);

            addpath(genpath(hmmLibPath));
            lambdas = trainTiedMixLeftrightHMM(dicProcDataTrTr, nfoTrTr, params, verbose); 
            [preds.categories, preds.loglikes] = testTiedMixLeftrightHMM(lambdas, dicProcDataVal);
            rmpath(genpath(hmmLibPath)); % interfieres with MATLAB functions (e.g. princomp)

            valModels{i,j}.params    = params;
            valModels{i,j}.lambdas   = lambdas;
            valModels{i,j}.preds     = preds;
            valAccuracies(i,j) = accuracy(nfoVal.categories, preds.categories);
        end
    end
    elapsedTime = toc(TSTART); 
    disp(sprintf('It took %.2f s.', elapsedTime));
        
    trainFoldOutput.classifier                          = classifierName;
    trainFoldOutput.scheme                              = scheme;
    trainFoldOutput.categories                          = categoriesInExpt;
    trainFoldOutput.params                              = params;
    trainFoldOutput.valParams.tiedMixParams.numMixtures = tiedMixParams.numMixtures;
    trainFoldOutput.numValidationFolds                  = numValidationFolds;
    
    validations.valModels = valModels;
    validations.valAccuracies = valAccuracies;
    trainFoldOutput.validations                         = validations;
    
    filepath = sprintf('%s/F%d.mat', outputDir, t);
    save(filepath, 'trainFoldOutput');
    
    % Remove temporary files
    % delete([tmpDir, '/*.mat']);
end


%% DICHOTOMIES' MODEL SELECTION

% disp('Model selection...');
% 
% P = zeros(size(C,1), numValidationFolds * length(subjects), size(D,2));
% for i = 1:length(subjects) 
%     s = subjects(i);
% 
%     filepathTr = sprintf('%s/F%d.mat', outputDir, s);
%     load(filepathTr, 'trainFoldOutput');
% 
%     for d = 1:size(D,2)
%         P(:,(i-1)*numValidationFolds+1:i*numValidationFolds,d) = ...
%             trainFoldOutput.dicValidations{d}.valAccuracies;
%     end
% end
% 
% bestC = zeros(2, size(D,2));
% for d = 1:size(D,2)
%     [maxval, maxidx] = max(mean(P(:,:,d),2));
%     bestC(:,d) = C(maxidx,:);
% end


%% DICHOTOMIES' FINAL MODELS CONSTRUCTION

% disp('Final selected models training...');
% 
% for i = 1:length(subjects)  
%     s = subjects(i);
%     
%     % Training data and info
%     dataTr           = data(nfo.subjects  ~= s);
%     nfoTr.categories = nfo.categories(nfo.subjects ~= s);
%     nfoTr.subjects   = nfo.subjects(nfo.subjects ~= s);
% 
%     filepathTr = sprintf('%s/F%d.mat', outputDir, s);
%     load(filepathTr, 'trainFoldOutput');
%     
%     teModels = cell(1,size(D,2));
%     % Train the classifiers using the best combination of numMixtures
%     for d = 1:size(D,2)
%         dicIndsTr = D(inds(nfoTr.categories),d)'; % double LUT(-ing)
%         
%         dicDataTr           = [dataTr(dicIndsTr == 1), dataTr(dicIndsTr == 2)];
%         dicNFOTr.categories = [dicIndsTr(dicIndsTr == 1),  dicIndsTr(dicIndsTr == 2)];
%         dicNFOTr.subjects   = [nfoTr.subjects(dicIndsTr == 1), nfoTr.subjects(dicIndsTr == 2)];
%         
%         [dicProcDataTr, params.preprocParams] = preprocessData(dicDataTr, params.preprocParams);
%         
%         params.tiedMixParams.numMixtures = bestC(:,d); % only validating this param
%         addpath(genpath(hmmLibPath));
%         lambdas = trainTiedMixLeftrightHMM(dicProcDataTr, dicNFOTr, params, verbose); 
%         rmpath(genpath(hmmLibPath)); % interfieres with MATLAB functions (e.g. princomp)
%         
%         teModels{d}.params = params;
%         teModels{d}.lambdas = lambdas;
%     end
%     
%     disp(sprintf('%d-th model trained.', i));
%     
%     trainFoldOutput.models = teModels;
%     save(filepathTr, 'trainFoldOutput');
% end


%% TESTING

% for i = 1:length(subjects)
%     s = subjects(i);
%     
%     % Test data and info
%     dataTe           = data(nfo.subjects == s);
%     nfoTe.categories = nfo.categories(nfo.subjects == s);
%     nfoTe.subjects   = nfo.subjects(nfo.subjects == s);
%     
%     filepathTr = sprintf('%s/F%d.mat', outputDir, s);
%     load(filepathTr, 'trainFoldOutput');
%         
%     predictions.metacategories = zeros(length(dataTe), size(D,2));
%     % Train the classifiers using the best combination of numMixtures
%     
%     for d = 1:size(D,2)
%         [procDataTe] = preprocessData([], trainFoldOutput.models{d}.params.preprocParams, dataTe);
% 
%         addpath(genpath(hmmLibPath));
%         [predictions.metacategories(:,d)] = ...
%             testTiedMixLeftrightHMM(trainFoldOutput.models{d}.lambdas, procDataTe); 
%         rmpath(genpath(hmmLibPath)); % interfieres with MATLAB functions (e.g. princomp)
%     end
%     
%     predictions.metacategories(predictions.metacategories == 2) = -1;
%     D(D == 2) = -1;
%     
%     S = pdist2(predictions.metacategories, D, distmetric);
%     [minVals, minInds] = min(S, [], 2);
%     predictions.categories = categoriesInExpt(minInds);
%     accs = accuracy(nfoTe.categories, predictions.categories)
%     
%     testFoldOutput.classifierName                      = classifierName;
%     testFoldOutput.scheme                              = scheme;
%     testFoldOutput.D                                   = D;
%     testFoldOutput.distmetric                          = distmetric;
%     testFoldOutput.params                              = params;
%     testFoldOutput.valParams.tiedMixParams.numMixtures = tiedMixParams.numMixtures;
%     testFoldOutput.numValidationFolds                  = numValidationFolds;
%     testFoldOutput.predictions                         = predictions;
%     testFoldOutput.accuracy                            = accs ;
%     
%     filepathTe = sprintf('%s/P%d.mat', outputDir, s);
%     save(filepathTe, 'testFoldOutput');
% end
% 
% outsampleAccs = zeros(1, length(subjects));
% for i = 1:length(subjects)
%     filepathTe = sprintf('%s/P%d.mat', outputDir, subjects(i));
%     load(filepathTe, 'testFoldOutput');
%     
%     outsampleAccs(i) = testFoldOutput.accuracy;
% end
% m = mean(outsampleAccs)
% ci = 2.26 * std(outsampleAccs)/sqrt(length(subjects))
