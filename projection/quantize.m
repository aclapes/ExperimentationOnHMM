function [ disDataTr, disDataTe ] = quantize( dataTr, dataTe, numVectors )
%PREPROCESS Summary of this function goes here
%   Detailed explanation goes here

    %% Dimensionality reduction

    % Standardize data
    XTr = cell2mat(dataTr);
    [ZTr, mu, sigma] = zscore(XTr);
    XTe = cell2mat(dataTe);
    ZTe = bsxfun(@times, bsxfun(@minus, XTe, mu), 1./sigma);
    ZTe(isnan(ZTe)) = .0;
    % Get the principal components space
    [COEFF,SCORE,latent] = princomp(ZTr);
    % Get those with associated greater eigenvalue (up to 90% of variance)
    q = 1:sum(cumsum(latent)/sum(latent) < 0.9) + 1;
    E = COEFF(:,q);
    % Project to that space (the one of Q eigenvectors)
    dataRedTr = ZTr * E;
    dataRedTe = ZTe * E;

    %% Vector quantization (K-means)

    rng(74, 'v5uniform');

    display(['Quantizing the data with ', num2str(numVectors), ' quantization vectors .. ']); 

    [indTr,C] = kmeans(dataRedTr,numVectors, ...
        'Distance','sqeuclidean',...
        'Start','cluster', ...
        'OnlinePhase', 'off', ...
        'EmptyAction', 'drop');

    S = pdist2(dataRedTe,C,'euclidean');
    [minVal, minIdx] = min(S,[],2);
    indTe = minIdx; % nearest centroid

    disDataTr = cell(length(dataTr),1); % Discrete observations
    c = 0;
    for j = 1:length(dataTr)
        n = size(dataTr{j},1);
        disDataTr{j} = indTr(c+1:c+n,:)';
        c = c + n;
    end

    disDataTe = cell(length(dataTe),1);
    c = 0;
    for j = 1:length(dataTe)
        n = size(dataTe{j},1);
        disDataTe{j} = indTe(c+1:c+n,:)';
        c = c + n;
    end
    
end

