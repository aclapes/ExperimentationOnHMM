close all;

%% General parametrization and data loading
parametrize;
load('data.mat');
load('12222122222222222222.mat');
labels = dicnfo(1,:);

%% Some numerical results

classes = unique(lut);
classesAccs = nanmean(outsampleAccs,2);
classesStds = nanstd(outsampleAccs,[],2);

% Outsample accuracy (LOSOCV)
display(['Weighted accuracy: ', ...
    num2str( nanmean(nanmean(outsampleAccs)) ), ' (', ...
    num2str(  nanstd(nanmean(outsampleAccs)) ), ')'] );
% Per-class outsample accuracy
bar(classes, classesAccs, 'y'); 
hold on; 
errorbar(classes, classesAccs, 2.26 * classesStds/sqrt(10), 2.26 * classesStds/sqrt(10),'b.'); 
hold on; 
ylim([0 1]); xlim([.5 length(classes)+.5]); 
xlabel('class'); ylabel('accuracy'); 
title('Per-class accuracies');


%% Visualizaton parametrization


%% Hidden aggregation 

% visData = hiddenAggregation(data, paths, numHidStates);
flen = 25;
visData = twoLvlPrincompAggregation(data, 0.5, flen);

% Get the principal components space
zisData = zscore(visData);
[COEFF,SCORE,LATENT] = princomp(zisData);

classes = unique(preds);
colors = [1 0 0; 
          0 0 1];

% Visualize 3-D
E = COEFF(:,1:3);
hidRata = zisData * E;
figure(2);
subplot(1,3,1);
for i = 1:length(classes)
    scatter3(hidRata(labels==i,1), ...
        hidRata(labels==i,2), ...
        hidRata(labels==i,3), ...
        10, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', colors(i,:));
    xlabel('x'); ylabel('y'); %zlabel('z');
    hold on;
end
title('Labels');
hold off;

subplot(1,3,2);
for i = 1:length(classes)
    scatter3(hidRata(preds==i,1), ...
        hidRata(preds==i,2), ...
        hidRata(preds==i,3), ...
        10, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', colors(i,:));
    xlabel('x'); ylabel('y'); zlabel('z');
    hold on;
end
title('Prediction');
hold off;

subplot(1,3,3);
for i = 1:length(classes)
    scatter3(hidRata(labels==i & preds==labels,1), ...
        hidRata(labels==i & preds==labels,2), ...
        hidRata(labels==i & preds==labels,3), ...
        10, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', [0 .75 0]);
    xlabel('x'); ylabel('y'); zlabel('z');
    hold on;
    scatter3(hidRata(labels==i & preds~=labels,1), ...
        hidRata(labels==i & preds~=labels,2), ...
        hidRata(labels==i & preds~=labels,3), ...
        10, 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', [.75 0 0]);
    xlabel('x'); ylabel('y'); zlabel('z');
    hold on;
end
title('Evaluated prediction');
hold off;


% Print the minority class more clearly

figure(3);
[minVal, minIdx] = min([sum(labels==1),sum(labels==2)]);

subplot(1,2,1);
scatter3(hidRata(labels==minIdx & preds==labels,1), ...
    hidRata(labels==minIdx & preds==labels,2), ...
    hidRata(labels==minIdx & preds==labels,3), ...
    'MarkerFaceColor', colors(minIdx,:), 'MarkerEdgeColor', [0 .75 0]);
xlabel('x'); ylabel('y'); zlabel('z');
hold on;
scatter3(hidRata(labels==minIdx & preds~=labels,1), ...
    hidRata(labels==minIdx & preds~=labels,2), ...
    hidRata(labels==minIdx & preds~=labels,3), ...
    'MarkerFaceColor', colors(minIdx,:), 'MarkerEdgeColor', [.75 0 0]);
xlabel('x'); ylabel('y'); zlabel('z');
hold on;

[maxVal, maxIdx] = max([sum(labels==1),sum(labels==2)]);

subplot(1,2,2);
scatter3(hidRata(labels==maxIdx & preds==labels,1), ...
    hidRata(labels==maxIdx & preds==labels,2), ...
    hidRata(labels==maxIdx & preds==labels,3), ...
    'MarkerFaceColor', colors(maxIdx,:), 'MarkerEdgeColor', [0 .75 0]);
xlabel('x'); ylabel('y'); zlabel('z');
hold on;
scatter3(hidRata(labels==maxIdx & preds~=labels,1), ...
    hidRata(labels==maxIdx & preds~=labels,2), ...
    hidRata(labels==maxIdx & preds~=labels,3), ...
    'MarkerFaceColor', colors(maxIdx,:), 'MarkerEdgeColor', [.75 0 0]);
xlabel('x'); ylabel('y'); zlabel('z');
hold on;
