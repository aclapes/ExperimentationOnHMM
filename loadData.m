function [ data, nfo ] = loadData( path, categories, subjects, examples, useConfidences )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here

nfo.categories = []; % Action info
nfo.subjects = [];
nfo.exampleIDs = [];
data = {}; % Action data

for i = 1:length(categories)
    c = categories(i);
    for j = 1:length(subjects)
        s = subjects(j);
        for k = 1:length(examples)
            e = examples(k);
            % Get the indexed skeleton
            skeleton = getskt3d(path, c, s, e, useConfidences);
%             isempty(skeleton)
            if ~isempty(skeleton)
                % Append to the data cell
                nfo.categories     = [nfo.categories, c];
                nfo.subjects    = [nfo.subjects, s];
                nfo.exampleIDs  = [nfo.exampleIDs, e];
                data{1,end+1}   = skeleton;
            end
        end
    end
end

end

