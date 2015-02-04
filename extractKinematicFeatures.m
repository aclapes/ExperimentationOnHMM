function dataFts = extractKinematicFeatures( data, velOffset )
%EXTRACTFEATURES Summary of this function goes here
%   Detailed explanation goes here

% Normalizing joint index
normJointIdx = 7; % pelvic joint
normJointInds = false(size(data{1},1),1); % logical indexing for norm joint rows indication  
normJointInds((3*(normJointIdx-1)+1):3*(normJointIdx)) = true;

for i = 1:length(data)
    seq = data{i}; % Time as rows and features as cols
    
    % Get normalized positions (relative to neck joint)
    normJointSeq = seq(normJointInds, :);
    normSktSeq = seq - repmat(normJointSeq,20,1);
    
    % Wrist-elbow-shoulder left and right angles
    leftElbowAngles = anglesBetweenJoints(normSktSeq, 1, 8, 10);
    rightElbowAngles = anglesBetweenJoints(normSktSeq, 2, 9, 11);
    leftShoulderAngles = anglesBetweenJoints(normSktSeq, 8, 1, 3);
    rightShoulderAngles = anglesBetweenJoints(normSktSeq, 9, 2, 3);
    
    % Get velocities
    velocities = velocitiesInJoints(normSktSeq, velOffset);
    
    % Build a joint representation (early feature fusion)
    seq = [normSktSeq(~normJointInds,:); velocities; ...
        leftElbowAngles; rightElbowAngles; leftShoulderAngles; rightShoulderAngles];
    seq(seq == 0) = 10e-4;
    dataFts{i} = seq;
end

end

