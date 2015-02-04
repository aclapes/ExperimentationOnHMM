function paths = mhmm_path(data, prior, transmat, mu, Sigma, mixmat)
% MHMM_PATH Compute hidden states path of sequences using Viterbi
% [loglik, errors] = mhmm_path(data, prior, transmat, mu, sigma, mixmat)
%
% data{m}(:,t) or data(:,t,m) if all cases have same length
% 
%
% Set mixmat to ones(Q,1) or omit it if there is only 1 mixture component

Q = length(prior);
if size(mixmat,1) ~= Q % trap old syntax
  error('mixmat should be QxM')
end
if nargin < 6, mixmat = ones(Q,1); end

if ~iscell(data)
  data = num2cell(data, [1 2]); % each elt of the 3rd dim gets its own cell
end
ncases = length(data);

paths = cell(ncases);
for m=1:ncases
  obslik = mixgauss_prob(data{m}, mu, Sigma, mixmat);
  paths{m} = viterbi_path(prior, transmat, obslik);
end
