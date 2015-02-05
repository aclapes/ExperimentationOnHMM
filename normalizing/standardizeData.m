function [stdData, m_, v_] = standardizeData( data, scale, m, v )
%STANDARDIZEDATA Summary of this function goes here
%   Detailed explanation goes here


D = cell2mat(data);

if nargin < 3
    [~, m, v] = zscore(D,0,2);
    m_ = m;
    v_ = v;
end

C = bsxfun(@minus, D, m);
Z = bsxfun(@times, C, 1./(scale.*v));

stdData = cell(size(data));
c = 0;
for i = 1:length(data)
    n = size(data{i},2);
    stdData{i} = Z(:,c+1:c+n);

    c = c + n;
end

end

