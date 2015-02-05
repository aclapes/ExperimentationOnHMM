function [ rata, E ] = dimreduction( data, varthresh, E )
%DIMREDUCTION Summary of this function goes here
%   Detailed explanation goes here

% Format the data
X = cell2mat(data)';

if nargin < 3
    % Get the principal components space
    [COEFF,SCORE,LATENT] = princomp(X);
    % Get those with associated greater eigenvalue (up to 90% of variance)
    q = 1:sum(cumsum(LATENT)/sum(LATENT) < varthresh) + 1;
    E = COEFF(:,q);
end

% Project to that space (the one of Q eigenvectors)
R = X * E;

rata = cell(size(data)); % Discrete observations
c = 0;
for j = 1:length(data)
    n = size(data{j},2);
    rata{j} = R(c+1:c+n,:)';
    c = c + n;
end

end

