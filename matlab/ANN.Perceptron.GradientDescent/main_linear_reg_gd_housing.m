TRAIN_DATA = load('e:\\study\\ml\\housing_train.txt');
[TRAIN_ROW_COUNT, TRAIN_COL_COUNT]  = size(TRAIN_DATA);

TEST_DATA = load('e:\\study\\ml\\housing_test.txt');
[TEST_ROW_COUNT, TEST_COL_COUNT] = size (TEST_DATA);

WHOLE_DATA = [TRAIN_DATA ;TEST_DATA ];
%normalize WHOLE_DATA 
WHOLE_DATA = normalize(WHOLE_DATA );

%restore train and test dat
TRAIN_DATA  = WHOLE_DATA(1:TRAIN_ROW_COUNT,:);
TEST_DATA =WHOLE_DATA(TRAIN_ROW_COUNT +1:end,:);

trainLabel = TRAIN_DATA(:,end); 
testLabel = TEST_DATA(:,end);

freeTerms = ones(TRAIN_ROW_COUNT,1); % free terms
TRAIN_DATA = [freeTerms,TRAIN_DATA]; % add free terms
TRAIN_DATA = TRAIN_DATA(:,1:TRAIN_COL_COUNT ); % discard lables

%initial weight vector w
w = rand(TRAIN_COL_COUNT,1) - 0.5; %% (-0.5, 0.5)
w(1) =1; % weight for the free term.

%jump into GD.!!!
w = linear_gd_batch_update(0.01, w, TRAIN_DATA, trainLabel, 1000);
out = TRAIN_DATA * w; 
MSEtrain = mean( (out - trainLabel) .^2)


free2 = ones(TEST_ROW_COUNT,1);
TEST_DATA = [free2,TEST_DATA]; % add free terms
TEST_DATA = TEST_DATA(:,1:TEST_COL_COUNT ); % discard lables
%size(w)
%size(TEST_DATA)
out = TEST_DATA * w;  

MSEtest = mean( (out - testLabel) .^2)