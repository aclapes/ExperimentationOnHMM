function results = validateTiedMixLeftrightHMM( data, nfo, numHidStates, ...
    selfTransProb, numMixtures, covType, maxIter, standardization, verbose)
%validedTiedMixLeftrightHMM Uses a LOSOCV to validate a tied mix leftright
%HMM with the indicated parameters:
%   numHidStates
%   selfTransProb
%   numMixtures
%   covType
%   maxIter

classes  = unique(nfo(1,:));
subjects = unique(nfo(2,:));

% Prepare testing results
preds = zeros(1,size(nfo,2));
likes = cell(1,size(nfo,2));
paths = cell(1,size(nfo,2));
outsampleAccs = nan( length(classes), length(subjects) );

% Loop
for i = 1:length(subjects)
    u = subjects(i);

    indicesTr = ( nfo(2,:) ~= u ); % leave i-th subject out
    indicesTe = ( nfo(2,:) == u );

    display(['Sbj ', num2str(i), '. Total data: ', num2str(length(data)), ', (', ...
        num2str(sum(indicesTr)/length(data)), '% train, ', num2str(sum(indicesTe)/length(data)), '% test).']);
    
    dataTr = data(indicesTr);
    dataTe = data(indicesTe);
    
    if standardization
        [dataTr, M, V] = standardizeData(dataTr);
        dataTe = standardizeData(dataTe, M, V);
    end
    
    rng(74);
    [predsTe, likesTe, pathsTe] = predictTiedMixLeftrightHMM( ...
                dataTr, dataTe, ...
                nfo(:,indicesTr), nfo(:,indicesTe), ...
                numHidStates, selfTransProb, numMixtures, covType, maxIter, verbose);

    preds(indicesTe) = predsTe;
    likes(indicesTe) = {likesTe};
    paths(indicesTe) = pathsTe;

    [~, A, C] = accuracy(nfo(1,indicesTe), predsTe);
    display(mean(A));
    
    outsampleAccs(C,i) = A;
end

% Parameters kept for results comprehension
params.numHidStates = numHidStates;
params.selfTransProb = selfTransProb;
params.numMixtures = numMixtures;
params.covType = covType;
params.maxIter = maxIter;

% Results comprehension
results.params = params;
results.nfo = nfo;
results.preds = preds;
results.likes = likes;
results.paths = paths;
results.outsampleAccs = outsampleAccs;

end

