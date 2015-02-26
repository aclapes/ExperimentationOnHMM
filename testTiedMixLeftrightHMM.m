function [preds, loglikes, paths] = testTiedMixLeftrightHMM( lambdas, data )
%TESTTIEDMIXLEFTRIGHTHMM Test models lambda predicting on data using the
% mAP estimate criterion.
% 
%   [preds, loglikes, paths] = testTiedMixLeftrightHMM( lambdas, data )
%

preds = zeros(1, size(data,2));
loglikes = zeros(length(lambdas), size(data,2));
if nargout > 2
    paths = cell(length(lambdas), size(data,2));
end

classes = 1:length(lambdas);

for i = 1:size(data,2)
    % Test on the different models
    for m = 1:length(lambdas)
        loglikes(m,i) = mhmm_logprob(data{i}, lambdas{m}.Pi, lambdas{m}.A, ...
            lambdas{m}.mu, lambdas{m}.Sigma, lambdas{m}.mixmat);
        if nargout > 2
            paths(m,i) = mhmm_path(data{i}, lambdas{m}.Pi, lambdas{m}.A, ...
                lambdas{m}.mu, lambdas{m}.Sigma, lambdas{m}.mixmat);
        end
    end
    % Prediction is mAP estimate
    [~, mAP] = max(loglikes(:,i)); % mAP
    preds(i) = classes(mAP);
end

varargout{1} = preds;
varargout{2} = loglikes;
if nargout > 2
    varargout{3} = paths;
end

end

