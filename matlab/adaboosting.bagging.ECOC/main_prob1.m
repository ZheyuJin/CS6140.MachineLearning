%% initial data loading
D = load('e:\\study\\ml\\spam.txt');
D = D(randperm(size(D,1)),:); % shuffle. yes.

%% runs with getOptStump function. 
adaboost( D, 15, @getOptStump);

%% runs with getOptStump function. 
adaboost( D, 300, @getRandStump);
