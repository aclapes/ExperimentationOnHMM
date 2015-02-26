function varargout = preprocessData( dataTr, params, varargin )
%PREPROCESSDATA Summary of this function goes here
%   Detailed explanation goes here

if nargin > 2
    dataTe = varargin{1};
end
%
% Data normalisation/scaling
%
normType = params.tiedMixParams.normParams(1,1); % 1st arg is scaling type
if normType > 0
    if normType == 1
        [dataTr, min, max] = minmaxData(dataTr);
        if nargin > 2
            dataTe = minmaxData(dataTe, min, max);
        end
    elseif normType == 2
        scale = params.tiedMixParams.normParams(1,2); % 2nd norm param is scaling type-dependent
        [dataTr, M, V] = standardizeData(dataTr, scale);
        if nargin > 2
            dataTe = standardizeData(dataTe, scale, M, V);
        end
    elseif normType == 3
        p = params.tiedMixParams.normParams(1,2); % scaling type-dependent
        dataTr = unitarizeData(dataTr, p);
        if nargin > 2
            dataTe = unitarizeData(dataTe, p);
        end
    end
end

%
% Data projection/dim.reduction
%

if params.tiedMixParams.projVar > 0
    [dataTr, E] = dimreduction(dataTr, params.tiedMixParams.projVar);
    if nargin > 2
        dataTe = dimreduction(dataTe, E);
    end
end

varargout{1} = dataTr;
if nargin > 2
    varargout{end+1} = dataTe;
end

end

