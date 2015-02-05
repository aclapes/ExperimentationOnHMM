function [stdData] = unitarizeData( data, p )
%UNITARIZEDATA Summary of this function goes here
%   Detailed explanation goes here


D = cell2mat(data);

N = sum(abs(D).^p,1).^(1/p);
U = bsxfun(@times, D, 1./N);
U(isnan(U)) = 0;

stdData = cell(size(data));
c = 0;
for i = 1:length(data)
    n = size(data{i},2);
    stdData{i} = U(:,c+1:c+n);

    c = c + n;
end

end

