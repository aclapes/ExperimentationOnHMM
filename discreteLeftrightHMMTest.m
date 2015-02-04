function acc = discreteLeftrightHMMTest( obsTr, obsTe, infoTr, infoTe, ...
    numObs, numHidStates, selfTransProb, maxIters)
%discreteLeftrightHMMTest Test the performance of a discrete left-right HMM classifying data.
%       data - A cell array of matrices representing NxP temporal sequences. 
%   N time instances and P features.
%       info - An colwise matrix representing the action, subject, and example
%   instance respectively.
%       numObs - # of possible observations of emission symbols in the HMM.
%       numHidStates - # of hidden states to model.
%       selfTransProb - a HMM parameter, which determines the probability of
%   self transitioning. Keyrole in LR models.
%       maxIters - HMM's regularisation parameter.


    %% Discrete leftright HMM (trained via EM)

    addpath(genpath('../Libs/HMMall/'));

    lambda0.Pi = normalise([1; zeros(numHidStates-1,1)]);
    lambda0.A = mk_leftright_transmat(numHidStates,selfTransProb);
    lambda0.B = mk_stochastic(rand(numHidStates,numObs));

    actionsTr = unique(infoTr(:,1));
    M = cell(length(actionsTr),1);
    for m = 1:length(actionsTr)
        seqsTr = obsTr(infoTr(:,1) == actionsTr(m));
        [~, M{m}.Pi, M{m}.A, M{m}.B] = dhmm_em(seqsTr, lambda0.Pi, lambda0.A, lambda0.B, 'max_iter', maxIters, 'verbose', false);
    end

    scoresTe = zeros(length(infoTe),1); %zeros(length(ATe),length(M));
    for j = 1:length(infoTe)
        for m = 1:length(M)
            scoresTe(j,m) = dhmm_logprob(obsTe{j}, M{m}.Pi, M{m}.A, M{m}.B);
        end
    end
    
    rmpath(genpath('../Libs/HMMall/'));

    %% Test the success
    
    [maxVal, maxIdx] = max(scoresTe,[],2);
    acc = (sum(maxIdx == infoTe(:,1))) / length(infoTe);

end

