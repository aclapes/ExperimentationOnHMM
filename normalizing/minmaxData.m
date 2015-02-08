function [ stdData, minVals_, maxVals_ ] = minmaxData( data, minVals, maxVals )
%MINMAXDATA Summary of this function goes here
%   Detailed explanation goes here

D = cell2mat(data);

if nargin < 3
    M = minmax(D);
    minVals = M(:,1);
    maxVals = M(:,2);
end

Z = bsxfun(@minus, D, minVals);
N = bsxfun(@times, Z, 1./(maxVals - minVals));

stdData = cell(size(data));
c = 0;
for i = 1:length(data)
    n = size(data{i},2);
    stdData{i} = N(:,c+1:c+n);

    c = c + n;
end

minVals_ = minVals;
maxVals_ = maxVals;

end



