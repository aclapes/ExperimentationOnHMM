function [ data, nfo ] = loadData( path, actions, subjects, examples, useConfidences )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here

nfo = []; % Action info
data = {}; % Action data

for i = 1:length(actions)
    a = actions(i);
    for j = 1:length(subjects)
        s = subjects(j);
        for k = 1:length(examples)
            e = examples(k);
            % Get the indexed skeleton
            skeleton = getskt3d(path, a, s, e, useConfidences);
%             isempty(skeleton)
            if ~isempty(skeleton)
                % Append to the data cell
                nfo = [nfo, [a;s;e]];
                data{1,end+1} = skeleton;
            end
        end
    end
end

end

