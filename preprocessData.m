function varargout = preprocessData( dataTr, params, varargin )
%PREPROCESSDATA Summary of this function goes here
%   Detailed explanation goes here

if nargin > 2
    dataTe = varargin{1};
end
%
% Data normalisation/scaling
%
normType = params.normParams(1,1); % 1st arg is scaling type
if normType > 0
    if normType == 1
        if ~isempty(dataTr)
            [dataTr, params.normDist.min, params.normDist.max] = minmaxData(dataTr);
        end
        if nargin > 2
            dataTe = minmaxData(dataTe, params.normDist.min, params.normDist.max);
        end
    elseif normType == 2
        scale = params.normParams(1,2); % 2nd norm param is scaling type-dependent
        if ~isempty(dataTr)
            [dataTr, params.normDist.M, params.normDist.V] = standardizeData(dataTr, scale);
        end
        if nargin > 2
            dataTe = standardizeData(dataTe, scale, params.normDist.M, params.normDist.V);
        end
    elseif normType == 3
        p = params.normParams(1,2); % scaling type-dependent
        dataTr = unitarizeData(dataTr, p);
        if nargin > 2
            dataTe = unitarizeData(dataTe, p);
        end
    end
end

%
% Data projection/dim.reduction
%

if params.projVar > 0
    if ~isempty(dataTr)
        [dataTr, params.projMat] = dimreduction(dataTr, params.projVar);
    end
    if nargin > 2
        dataTe = dimreduction(dataTe, params.projMat);
    end
end

if ~isempty(dataTr)
    varargout{1} = dataTr;
    varargout{2} = params;
    if nargin > 2
        varargout{end+1} = dataTe;
    end
else
    varargout{1} = dataTe;
end


end

