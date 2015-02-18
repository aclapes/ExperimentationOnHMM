function [predsTe, loglikesTe, pathsTe] = predictTiedMixLeftrightHMM( obsTr, obsTe, infoTr, infoTe, ...
    numHidStates, selfTransProb, numMixtures, emInit, covType, maxIters, verbose)
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
    
    classes = unique(infoTr(1,:));
    lambdas = cell(length(classes),1);
    
    time = 0;
    
    for m = 1:length(classes)
        seqsTr = obsTr(infoTr(1,:) == classes(m));
        
        seqsTrSrl = cell2mat(seqsTr);
        O = size(seqsTrSrl,1);
        success = 0;
        tries = 0;
        TSTART = tic;
        while ~success
            try
                [lambda0.mu, lambda0.Sigma] = mixgauss_init(numHidStates(m)*numMixtures(m), seqsTrSrl, covType, emInit);
                success = 1;
            catch err
                tries = tries + 1;
                success = 0;
            end
        end
        t1 = toc(TSTART);
        
        lambda0.Pi = normalise([1; zeros(numHidStates(m)-1,1)]);
        lambda0.A = mk_leftright_transmat(numHidStates(m),selfTransProb);
        lambda0.mu = reshape(lambda0.mu, [O numHidStates(m) numMixtures(m)]);
        lambda0.Sigma = reshape(lambda0.Sigma, [O O numHidStates(m) numMixtures(m)]);
        lambda0.mixmat = mk_stochastic(rand(numHidStates(m),numMixtures(m)));
        
        TSTART = tic;
        [~, lambdas{m}.Pi, lambdas{m}.A, lambdas{m}.mu, lambdas{m}.Sigma, lambdas{m}.mixmat] = mhmm_em( ...
            seqsTr, lambda0.Pi, lambda0.A, ...
            lambda0.mu, lambda0.Sigma, lambda0.mixmat, ...
            'max_iter', maxIters, 'verbose', 0, 'cov_type', covType);
        t2 = toc(TSTART);
        
        mtime = t1 + t2;
        if verbose
            display(['Model ', num2str(m),' training took ', num2str(mtime), ...
                ' secs and ', num2str(tries), ' non-psds matrices.']);
         end
        time = time + mtime;
    end
    
    if verbose
        display(['All models training took ', num2str(time), ' secs.']);
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
        predsTe(j) = classes(mAP);
        % Predict the path
        pathsTe(j) = mhmm_path(obsTe{j}, lambdas{mAP}.Pi, lambdas{mAP}.A, ...
            lambdas{mAP}.mu, lambdas{mAP}.Sigma, lambdas{mAP}.mixmat);
    end
    
    rmpath(genpath('../Libs/HMMall/'));
end

