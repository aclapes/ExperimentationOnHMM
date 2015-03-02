% TestCA1.m - Test a class-aggregated continuous HMM

clear all;

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
distmetric = 'cityblock'; % distance metric
D = codingECOC(scheme, length(categoriesInExpt)); % ECOC dichotomies' coding matrix

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
aux = repmat({tiedMixParams.numMixtures}, 1, 2);
C = allcomb(aux{:}); % if using ECOC, it is numMixtures^2

% Output results will be kept in a directory named as this script
outputDir = ['output/results/', mfilename, scheme]; % mfilename returns the script name
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
    
    dicValidations = cell( 1, size(D,2) ); 

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
        
        % % train and keep models, that are {combinations x internal folds x
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
                [dicProcDataTrTr, ~, dicProcDataVal] = preprocessData(dicDataTrTr, params.preprocParams, dicDataVal);
                
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
        dicValidations{d} = dicValidation;
    end
     
    % Save the test fold's results in a file
    trainFoldOutput.classifier                          = classifierName;
    trainFoldOutput.scheme                              = scheme;
    trainFoldOutput.categories                          = categoriesInExpt;
    trainFoldOutput.D                                   = D;
    trainFoldOutput.params                              = params;
    trainFoldOutput.valParams.tiedMixParams.numMixtures = tiedMixParams.numMixtures;
    trainFoldOutput.numValidationFolds                  = numValidationFolds;
    trainFoldOutput.dicValidations                      = dicValidations;
    
    filepath = sprintf('%s/F%d.mat', outputDir, t);
    save(filepath, 'trainFoldOutput');
    
    % Remove temporary files
    % delete([tmpDir, '/*.mat']);
end


%% DICHOTOMIES' MODEL SELECTION

disp('Model selection...');

P = cell(1,size(D,2));
for i = 1:length(subjects) 
    s = subjects(i);

    filepathTr = sprintf('%s/F%d.mat', outputDir, s);
    load(filepathTr, 'trainFoldOutput');

     % Perhaps we want to constraint the num of mixtures despite having
    % computed more than tiedMixParams.numMixtures in the training
    aux = repmat({trainFoldOutput.valParams.tiedMixParams.numMixtures}, 1, 2);
    trainedC = allcomb(aux{:});
    constrIndices = logical(prod(trainedC <= tiedMixParams.numMixtures(end), 2));
    
    for d = 1:size(D,2)
        P{d} = [P{d}, trainFoldOutput.dicValidations{d}.valAccuracies(constrIndices,:)];
    end
end

bestC = zeros(2, size(D,2));
for d = 1:size(D,2)
    [maxval, maxidx] = max(mean(P{d},2));
    bestC(:,d) = C(maxidx,:);
end


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
%     teModels = cell(numFinalModelReplicas,size(D,2));
%     TSTART = tic;
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
%         for r = 1:numFinalModelReplicas
%             params.tiedMixParams.numMixtures = bestC(:,d); % only validating this param
%             addpath(genpath(hmmLibPath));
%             lambdas = trainTiedMixLeftrightHMM(dicProcDataTr, dicNFOTr, params, verbose); 
%             rmpath(genpath(hmmLibPath)); % interfieres with MATLAB functions (e.g. princomp)
% 
%             teModels{r,d}.params = params;
%             teModels{r,d}.lambdas = lambdas;
%         end
%     end
%     elapsedTime = toc(TSTART);
%     disp(sprintf('%d-th model trained. Took %.2f secs.', i, elapsedTime));
%     
%     trainFoldOutput.models = teModels;
%     save(filepathTr, 'trainFoldOutput');
% end


%% TESTING

for i = 1:length(subjects)
    s = subjects(i);
    
    % Test data and info
    dataTe           = data(nfo.subjects == s);
    nfoTe.categories = nfo.categories(nfo.subjects == s);
    nfoTe.subjects   = nfo.subjects(nfo.subjects == s);
    
    filepathTr = sprintf('%s/F%d.mat', outputDir, s);
    load(filepathTr, 'trainFoldOutput');
    
    trainFoldOutput.D(trainFoldOutput.D == 2) = -1;
    
    accs = zeros(size(trainFoldOutput.models, 1), 1);
    P = cell(size(trainFoldOutput.models, 1), 1);
    for r = 1:size(trainFoldOutput.models, 1)
        metacategories = zeros(length(dataTe), size(trainFoldOutput.D,2));

        % Train the classifiers using the best combination of numMixtures
        for d = 1:size(trainFoldOutput.models, 2)
            [procDataTe] = preprocessData([], trainFoldOutput.models{r,d}.params.preprocParams, dataTe);

            addpath(genpath(hmmLibPath));
            [metacategories(:,d)] = ...
                testTiedMixLeftrightHMM(trainFoldOutput.models{r,d}.lambdas, procDataTe); 
            rmpath(genpath(hmmLibPath)); % interfieres with MATLAB functions (e.g. princomp)
        end

        metacategories(metacategories == 2) = -1;

        S = pdist2(metacategories, trainFoldOutput.D, distmetric);
        [minVals, minInds] = min(S, [], 2);
        P{r}.categories = categoriesInExpt(minInds);
        accs(r) = accuracy(nfoTe.categories, P{r}.categories);
    end
    
    testFoldOutput.classifierName                      = trainFoldOutput.classifier;
    testFoldOutput.scheme                              = trainFoldOutput.scheme;
    testFoldOutput.D                                   = trainFoldOutput.D;
    testFoldOutput.params                              = trainFoldOutput.params;
    testFoldOutput.valParams                           = trainFoldOutput.valParams;
    testFoldOutput.numValidationFolds                  = trainFoldOutput.numValidationFolds;
    testFoldOutput.distmetric                          = distmetric;
    testFoldOutput.predictions                         = P;
    testFoldOutput.accs                                = accs ;
    
    filepathTe = sprintf('%s/P%d.mat', outputDir, s);
    save(filepathTe, 'testFoldOutput');
end

outsampleAccs = zeros(length(subjects), numFinalModelReplicas);
for i = 1:length(subjects)
    filepathTe = sprintf('%s/P%d.mat', outputDir, subjects(i));
    load(filepathTe, 'testFoldOutput');
    
    for r = 1:numFinalModelReplicas
        outsampleAccs(i,r) = testFoldOutput.accs(r);
    end
end
m = mean(outsampleAccs(:))
ci = 1.96 * std(outsampleAccs(:))/sqrt(numel(outsampleAccs))
