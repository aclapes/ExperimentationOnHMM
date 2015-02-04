function angles = angleBetweenJoints( poses, ja, jb, jc )
%ANGLEBETWEENJOINTS Calculates the angle forming the ba and bc vectors.
%   Uses the atan2 for better numerical stability.

    btoaVecs = poses((3*(ja-1)+1):3*(ja), :) ...
        - poses((3*(jb-1)+1):3*(jb), :);
    btocVecs = poses((3*(jc-1)+1):3*(jc), :) ...
        - poses((3*(jb-1)+1):3*(jb), :);
    
    term1 = cross(btoaVecs, btocVecs);
    term2 = dot(btoaVecs, btocVecs);
    normTerm1 = sqrt(sum(term1.^2));
    
    angles = atan2(normTerm1, term2);

end

