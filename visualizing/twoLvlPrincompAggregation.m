function prjData = twoLvlPrincompAggregation( data, varthresh, flen )
%TWOLVLPRINCOMPAGGREGATION Summary of this function goes here
%   Detailed explanation goes here

    D = cell2mat(data);

    % Standardize data
    [Z, mu, sigma] = zscore(D');
    
    % Get the principal components space
    [COEFF,SCORE,latent] = princomp(Z);
    
    % Get those with associated greater eigenvalue (up to 90% of variance)
    q = sum(cumsum(latent)/sum(latent) < varthresh) + 1;
    E = COEFF(:,1:q);

    R = Z * E;

    rataTr = cell(size(data)); % Discrete observations
    c = 0;
    for j = 1:length(data)
        n = size(data{j},2);
        V = R(c+1:c+n,:);
        Q = imresize(V,[q,flen]);
        
        rataTr{j} = reshape(Q,q*flen,1);
        c = c + n;
    end
    
    prjData = cell2mat(rataTr)';
end