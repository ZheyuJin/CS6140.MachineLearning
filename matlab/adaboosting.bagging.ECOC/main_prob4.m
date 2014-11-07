%% loading trainin and test data.
%D = read_matrix( '8newsgroup\train.trec\feature_matrix.txt',11314,1754 );%train
%T = read_matrix( '8newsgroup\test.trec\feature_matrix.txt',7532,1754 );%test

%% random ECOC with size 20.
FunctionCount = 20;
ECOC_Table = round(rand(8,FunctionCount)); % 8 class, 20 ECOC funcs.



%% tranning
tic
ROUND_MAX = 100;
getStumpFunc = @getOptStump_fast; %hope it's fast.
originalTestLabel = D(:,end);

M = zeros([size(D),FunctionCount]);
for f  = 1: FunctionCount
M(:,:,f)    = D;
end
toc

tic
parfor f  = 1: FunctionCount
    mapping = ECOC_Table(:,f);
    %mapping label.
    M(:,end,f) = mapper(mapping, originalTestLabel);
    
    % do boosing
    [ alphaVec,stumpVec] ...
    = adaboosing_training( D(:,1:end-1),D(:,end),ROUND_MAX,getStumpFunc);
    alphaVec
end
toc