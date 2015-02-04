close all;

%% General parametrization and data loading
parametrize;
load('data.mat');
load('22212112122111212112.mat');
labels = dicnfo(1,:);

%% Some numerical results

classes = unique(lut);
classesAccs = nanmean(outsampleAccs,2);
classesStds = nanstd(outsampleAccs,[],2);

% Outsample accuracy (LOSOCV)
display(['Weighted accuracy: ', ...
    num2str( nanmean(nanmean(outsampleAccs)) ), ' (', ...
    num2str(  nanstd(nanmean(outsampleAccs)) ), ')'] );
% % Per-class outsample accuracy
% bar(classes, classesAccs, 'y'); 
% hold on; 
% errorbar(classes, classesAccs, 2.26 * classesStds/sqrt(10), 2.26 * classesStds/sqrt(10),'b.'); 
% hold on; 
% ylim([0 1]); xlim([.5 length(classes)+.5]); 
% xlabel('class'); ylabel('accuracy'); 
% title('Per-class accuracies');


