function hidData = hiddenAggregation( data, paths, numHidStates )
%HIDDENAGGREGATION Summary of this function goes here
%   Detailed explanation goes here

numFeatures = size(data{1},1);
hidData = zeros(length(data), numHidStates * numFeatures);
for i = 1:length(data)
    seq = data{i};
    hidstates = paths{i};
    
    flseq = zeros(numFeatures,numHidStates);
    for j = 1:numHidStates
        center = mean(seq(:, hidstates == j),2);
        center(isnan(center)) = 0;
        variances = bsxfun(@minus, seq(:,hidstates == j), center);
        aggVars = sum(variances,2);
        aggVars(isnan(aggVars)) = 0;
        normAggVars = aggVars ./ norm(aggVars,2);
        normAggVars(isnan(normAggVars)) = 0;
        flseq(:,j) = normAggVars;
    end
    hidData(i,:) = reshape(flseq,1,numFeatures*numHidStates);
end

end

