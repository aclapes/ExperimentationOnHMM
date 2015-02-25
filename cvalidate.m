function CVO = cvalidate( varin, scheme, varargin )
%CODINGECOC Summary of this function goes here
%   Detailed explanation goes here

% if prod(size(varin)) == 1
%     N = varin;
% else
%     labels = varin;
     
CVO = {};

if strcmp(lower(scheme), 'kfold')

elseif strcmp(lower(scheme), 'leaveout')

elseif strcmp(lower(scheme), 'leaveclassout')
    [classes, aps] = unique(varin);
    CVO.NumTestSets = length(classes);

    p = randperm(length(classes)); % an indices' permutation
    inds(classes(p)) = 1:length(classes);
    
    for i = 1:length(classes)
        b = zeros(length(classes));
        b(i) = 1;
        
        CVO.test{i} = b(inds(varin));
        CVO.training{i} = ones(1,length(varin)) & ~CVO.test{i};
    end
end
        

end

