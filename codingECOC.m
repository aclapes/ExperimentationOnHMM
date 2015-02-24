function M = codingECOC( scheme, nclasses )
%CODINGECOC Summary of this function goes here
%   Detailed explanation goes here

    if strcmp(scheme, 'onevsall')
        M = -eye(nclasses) + 2;
    elseif strcmp(scheme, 'onevsone')
        M = zeros(nclasses,nchoosek(nclasses,2),'int8');
        
        C = flipud(combnk([1:nclasses],2));
        for i = 1:size(C,1)
            M(C(i,1),i) = 1;
            M(C(i,2),i) = 2;
        end
    end

end

