function [procDataTr, procDataTe] = preprocessData( dataTr, dataTe, params )
%PREPROCESSDATA Summary of this function goes here
%   Detailed explanation goes here

%
% Data normalisation/scaling
%

normType = params.tiedMixParams.normParams(1,1); % 1st arg is scaling type
if normType > 0
    if normType == 1
        [dataTr, min, max] = minmaxData(dataTr);
        dataTe = minmaxData(dataTe, min, max);
    elseif normType == 2
        scale = params.tiedMixParams.normParams(1,2); % 2nd norm param is scaling type-dependent
        [dataTr, M, V] = standardizeData(dataTr, scale);
        dataTe = standardizeData(dataTe, scale, M, V);
    elseif normType == 3
        p = params.tiedMixParams.normParams(1,2); % scaling type-dependent
        dataTr = unitarizeData(dataTr, p);
        dataTe = unitarizeData(dataTe, p);
    end
end

%
% Data projection/dim.reduction
%
 
if params.tiedMixParams.projVar > 0
    [dataTr, E] = dimreduction(dataTr, params.tiedMixParams.projVar);
    dataTe = dimreduction(dataTe, E);
end

procDataTr = dataTr;
procDataTe = dataTe;

end

