function velocities = velocitiesInJoints( poses, offset )
%VELOCITIESINJOINTS Compute the velocities in the joints. 

    displacements = [zeros(size(poses,1),offset), poses(:,1+offset:end) - poses(:,1:end-offset)];
    
    numJoints = size(poses,1)/3;
    
    velocities = zeros(numJoints, size(displacements,2));
    for j = 1:numJoints
        displacements_j = displacements((j-1)*3+1:j*3,:);
        velocities(j,:) = sqrt(sum(displacements_j.^2));
    end
end

