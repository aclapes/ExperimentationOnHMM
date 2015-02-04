function [acc, A, classes] = accuracy( labels, predictions )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

classes = unique(labels);

A = zeros(length(classes),1); % accuracies of the different classes
acc = 0;

for i = 1:length(classes)
    hits = sum(labels == classes(i) & predictions == labels);
    misses = sum(labels == classes(i) & predictions ~= labels);
    
    A(i) = hits / (hits + misses);
end
acc = mean(A);

end

