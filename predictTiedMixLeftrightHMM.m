function [predsTe, loglikesTe, pathsTe] = predictTiedMixLeftrightHMM( obsTr, obsTe, infoTr, infoTe, ...
    numHidStates, selfTransProb, numMixtures, covType, maxIters)
%continuousLeftrightHMMTest Test the performance of a continuous left-right HMM classifying data.
%       data - A cell array of matrices representing NxP temporal sequences. 
%   N time instances and P features.
%       info - An colwise matrix representing the action, subject, and example
%   instance respectively.
%       numHidStates - # of hidden states to model.
%       selfTransProb - a HMM parameter, which determines the probability of
%   self transitioning. Keyrole in LR models.
%       numMixtures - number of mixtures.
%       covType - Covariance matrix's type.
%       maxIters - HMM's regularisation parameter.

    %% Continuous Left-right HMM (trained via EM)

    addpath(genpath('../Libs/HMMall/'));

    lambda0.Pi = normalise([1; zeros(numHidStates-1,1)]);
    lambda0.A = mk_leftright_transmat(numHidStates,selfTransProb);
    
    actionsTr = unique(infoTr(1,:));
    lambdas = cell(length(actionsTr),1);
    for m = 1:length(actionsTr)
        seqsTr = obsTr(infoTr(1,:) == actionsTr(m));
        
        seqsTrSrl = cell2mat(seqsTr);
        O = size(seqsTrSrl,1);
        success = 0;
        while ~success
            try
                [lambda0.mu, lambda0.Sigma] = mixgauss_init(numHidStates*numMixtures(m), seqsTrSrl, 'diag');
                success = 1;
            catch err
                success = 0;
            end
        end
        lambda0.mu = reshape(lambda0.mu, [O numHidStates numMixtures(m)]);
        lambda0.Sigma = reshape(lambda0.Sigma, [O O numHidStates numMixtures(m)]);
        lambda0.mixmat = mk_stochastic(rand(numHidStates,numMixtures(m)));
        
        [~, lambdas{m}.Pi, lambdas{m}.A, lambdas{m}.mu, lambdas{m}.Sigma, lambdas{m}.mixmat] = mhmm_em( ...
            seqsTr, lambda0.Pi, lambda0.A, ...
            lambda0.mu, lambda0.Sigma, lambda0.mixmat, ...
            'max_iter', maxIters, 'verbose', 0, 'cov_type', covType);
    end

    predsTe = zeros(1,size(infoTe,2));
    loglikesTe = zeros(length(lambdas),size(infoTe,2));
    pathsTe = cell(1,size(infoTe,2));
    
    for j = 1:length(infoTe)
        % Test on the different HMM models
        for m = 1:length(lambdas)
            loglikesTe(m,j) = mhmm_logprob(obsTe{j}, lambdas{m}.Pi, lambdas{m}.A, ...
                lambdas{m}.mu, lambdas{m}.Sigma, lambdas{m}.mixmat);
        end
        % Prediction is mAP estimate
        [~, mAP] = max(loglikesTe(:,j)); % mAP
        predsTe(j) = mAP;
        % Predict the path
        pathsTe(j) = mhmm_path(obsTe{j}, lambdas{mAP}.Pi, lambdas{mAP}.A, ...
            lambdas{mAP}.mu, lambdas{mAP}.Sigma, lambdas{mAP}.mixmat);
    end
    
    rmpath(genpath('../Libs/HMMall/'));
end

