function lambdas = trainTiedMixLeftrightHMM( dataTr,  nfoTr, ...
    params, verbose)
%     numHidStates, selfTransProb, numMixtures, ...
%     normParams, projVar, emInit, covType, maxIter, verbose)
%validedTiedMixLeftrightHMM Validate a tied mix leftright using LOSOCV

classes = unique(nfoTr.categories);
lambdas = cell(1, length(classes));

if prod(size(params.numHidStates)) == 1
    params.numHidStates = repmat(params.numHidStates, 1, length(classes));
end
if prod(size(params.selfTransProb)) == 1
    params.selfTransProb = repmat(params.selfTransProb, 1, length(classes));
end

%
% Training of the HMM
%
   
for m = 1:length(classes)
    c = classes(m); % get the class label
    lambdas{m}.id = c;

    % filter the class data in training
    classDataTr = dataTr(nfoTr.categories == c); 

    % serialize horizontal class data for training
    classDataTrSrl = cell2mat(classDataTr); 
    O = size(classDataTrSrl,1); % num of observations in a class
    
    % build an initial mixture distribution (needed for the initial HMM
    % model)
    success = 0;
    while ~success
        try
            [model0.mu, model0.Sigma] = mixgauss_init( ...
                params.numHidStates(m)*params.tiedMixParams.numMixtures(m), ...
                classDataTrSrl, params.tiedMixParams.covType, params.tiedMixParams.emInit);
            success = 1;
        catch err
            success = 0;
        end
    end

    % create the initial HMM model: model0
    model0.Pi = normalise([1; zeros(params.numHidStates(m)-1,1)]);
    model0.A = mk_leftright_transmat(params.numHidStates(m),params.selfTransProb(m));
    model0.mu = reshape(model0.mu, [O params.numHidStates(m) params.tiedMixParams.numMixtures(m)]);
    model0.Sigma = reshape(model0.Sigma, [O O params.numHidStates(m) params.tiedMixParams.numMixtures(m)]);
    model0.mixmat = mk_stochastic(rand(params.numHidStates(m),params.tiedMixParams.numMixtures(m)));

    % actual training
    [~, lambdas{m}.Pi, lambdas{m}.A, lambdas{m}.mu, lambdas{m}.Sigma, lambdas{m}.mixmat] = mhmm_em( ...
        classDataTr, model0.Pi, model0.A, ...
        model0.mu, model0.Sigma, model0.mixmat, ...
        'max_iter', params.maxIter, 'verbose', 0, 'cov_type', params.tiedMixParams.covType);
end

% predsTe = zeros(1,size(infoTe,2));
% loglikesTe = zeros(length(lambdas),size(infoTe,2));
% pathsTe = cell(1,size(infoTe,2));
% 
% for j = 1:length(infoTe)
%     % Test on the different HMM models
%     for m = 1:length(lambdas)
%         loglikesTe(m,j) = mhmm_logprob(obsTe{j}, lambdas{m}.Pi, lambdas{m}.A, ...
%             lambdas{m}.mu, lambdas{m}.Sigma, lambdas{m}.mixmat);
%     end
%     % Prediction is mAP estimate
%     [~, mAP] = max(loglikesTe(:,j)); % mAP
%     predsTe(j) = classes(mAP);
%     % Predict the path
%     pathsTe(j) = mhmm_path(obsTe{j}, lambdas{mAP}.Pi, lambdas{mAP}.A, ...
%         lambdas{mAP}.mu, lambdas{mAP}.Sigma, lambdas{mAP}.mixmat);
% end

    
% %     rng(74);
%     [predsTe, models{i}, likesTe, pathsTe] = ...
%         predictTiedMixLeftrightHMM( dataTr, dataTe, ...
%         nfo(:,indicesTr), nfo(:,indicesTe), ...
%         numHidStates, selfTransProb, numMixtures, ...
%         emInit, covType, maxIter, verbose);
end

