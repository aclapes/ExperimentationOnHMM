G = [1:100]; % Number of groundtruth elements
P = [1:100]; % Number of predicted elements

%% The function (f) calculation

V = {G,P};
C = allcomb(V{:}); % Generate the 2-D input space of f

f = 1 ./ (1 + abs(C(:,1)-C(:,2)) );

%% Print the function
surf( reshape(f,length(G),length(P)) ); 
hold on;
title('Jaccvagina');
xlabel('G'); ylabel('P');
hold off;

mfilename