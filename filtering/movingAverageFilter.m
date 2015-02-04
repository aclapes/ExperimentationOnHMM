function data = movingAverageFilter( data, lag )
%MOVINGAVERAGEFILTER Summary of this function goes here
%   Detailed explanation goes here

% Smooth skeletons
for i = 1:length(data)
    seqt = data{i}'; % Seq transposed because tsmovavg later assumes rows
    dummy = repmat(seqt(1,:),lag,1);
    filtSktSeq = tsmovavg( [dummy; seqt],'e', lag, 1); % assumes row-wise
    data{i} = filtSktSeq((1+lag):end, :)';
end

end

