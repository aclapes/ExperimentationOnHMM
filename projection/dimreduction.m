function [ rataTr, rataTe ] = dimreduction( dataTr, dataTe, varthresh )
%DIMREDUCTION Summary of this function goes here
%   Detailed explanation goes here

    % Format the data
    XTr = cell2mat(dataTr)';
    XTe = cell2mat(dataTe)';
    
    % Standardize data
    [ZTr, mu, sigma] = zscore(XTr);
    ZTe = bsxfun(@times, bsxfun(@minus, XTe, mu), 1./sigma);
    ZTe(isnan(ZTe)) = .0;
    
    % Get the principal components space
    [COEFF,SCORE,latent] = princomp(ZTr);
    
    % Get those with associated greater eigenvalue (up to 90% of variance)
    q = 1:sum(cumsum(latent)/sum(latent) < varthresh) + 1;
    E = COEFF(:,q);
    % Project to that space (the one of Q eigenvectors)
    RTr = ZTr * E;
    RTe = ZTe * E;
    
    rataTr = cell(size(dataTr)); % Discrete observations
    c = 0;
    for j = 1:length(dataTr)
        n = size(dataTr{j},2);
        rataTr{j} = RTr(c+1:c+n,:)';
        c = c + n;
    end

    rataTe = cell(size(dataTe));
    c = 0;
    for j = 1:length(dataTe)
        n = size(dataTe{j},2);
        rataTe{j} = RTe(c+1:c+n,:)';
        c = c + n;
    end

end

