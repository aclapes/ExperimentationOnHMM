function results = validateTiedMixLeftrightHMM( data, nfo, ...
    numHidStates, selfTransProb, numMixtures, ...
    normParams, projVar, emInit, covType, maxIter, verbose)
%validedTiedMixLeftrightHMM Validate a tied mix leftright using LOSOCV

classes  = unique(nfo(1,:));
subjects = unique(nfo(2,:));

% Prepare testing results
preds = zeros(1,size(nfo,2));
likes = cell(1,size(nfo,2));
paths = cell(1,size(nfo,2));
outsampleAccs = nan( length(classes), length(subjects) );

models = cell(1,length(subjects));

% Loop
for i = 1:length(subjects)
    u = subjects(i);

    indicesTr = ( nfo(2,:) ~= u ); % leave i-th subject out
    indicesTe = ( nfo(2,:) == u );

    if verbose
        display(['Sbj ', num2str(i), '. Total data: ', num2str(length(data)), ', (', ...
        num2str(sum(indicesTr)/length(data)), '% train, ', num2str(sum(indicesTe)/length(data)), '% test).']);
    end
    
    dataTr = data(indicesTr);
    dataTe = data(indicesTe);
    
    normType = normParams(1,1);
    if normType > 0
        if normType == 1
            [dataTr, min, max] = minmaxData(dataTr);
            dataTe = minmaxData(dataTe, min, max);
        elseif normType == 2
            scale = normParams(1,2);
            [dataTr, M, V] = standardizeData(dataTr, scale);
            dataTe = standardizeData(dataTe, scale, M, V);
        elseif normType == 3
            p = normParams(1,2);
            dataTr = unitarizeData(dataTr, p);
            dataTe = unitarizeData(dataTe, p);
        end
    end
    
    if projVar > 0
        [dataTr, E] = dimreduction(dataTr, projVar);
        dataTe = dimreduction(dataTe, E);
    end
    
%     rng(74);
    [predsTe, models{i}, likesTe, pathsTe] = ...
            predictTiedMixLeftrightHMM( dataTr, dataTe, ...
                nfo(:,indicesTr), nfo(:,indicesTe), ...
                numHidStates, selfTransProb, numMixtures, ...
                emInit, covType, maxIter, verbose);

    preds(indicesTe) = predsTe;
    likes(indicesTe) = {likesTe};
    paths(indicesTe) = pathsTe;

    [~, A, C] = accuracy(nfo(1,indicesTe), predsTe);
    if verbose
        display(mean(A));
    end
    
    outsampleAccs(C,i) = A;
end

% Parameters kept for results comprehension
params.numHidStates = numHidStates;
params.selfTransProb = selfTransProb;
params.numMixtures = numMixtures;
params.normParams = normParams;
params.projVar = projVar;
params.emInit = emInit;
params.covType = covType;
params.maxIter = maxIter;

% Results comprehension
results.params = params;
results.nfo = nfo;
results.preds = preds;
results.likes = likes;
results.paths = paths;
results.outsampleAccs = outsampleAccs;
results.models = models;

end

