D = load('e:\\study\\ml\\spam.txt');
D = D(randperm(size(D,1)),:); % shuffle. yes.
[rows,cols] = size(D);

kfold = 10;
blockRowCount = int32(rows /kfold);

trainErrCount =0;
trainCorrectCount =0;

testErrCount =0;
testCorrectCount =0;

%preallocate train and test matrix
train = cell(rows - blockRowCount, cols);
test = cell(blockRowCount, cols);


for curFold  = 0 : kfold -1
    %data loading for train and test
    test = D(1+blockRowCount*curFold: blockRowCount*(curFold +1) , :);
    
    train = [D(1: blockRowCount*curFold ,:);
        D(blockRowCount*(curFold +1)+1: rows, : )];
    
    size(test);
    size(train)   ;
    
    
    label = train(:,cols:cols); % label
    [train_row, train_col] = size(train);
    free = ones(train_row,1); % free terms
    train = [free,train]; % add free terms
    train = train(:,1:cols ); % discard lables
    
    TRANS = transpose(train);
    w = inv(TRANS  * train) * TRANS  * label; %got it.
    
    %train MSE
    out = train * w;
    
    
    [tmpACC,tmpERR] = spam_acc_err_count(out , label);
    trainCorrectCount = trainCorrectCount + tmpACC;
    trainErrCount = trainErrCount + tmpERR;
    
    %test MSE
    label = test(:,cols:cols); % label
    [test_row, test_col] = size(test);
    free = ones(test_row,1); % free terms
    test = [free,test]; % add free terms
    test = test(:,1:cols ); % discard lables
    out = test * w;
    
    [tmpACC,tmpERR] = spam_acc_err_count(out , label);
    testCorrectCount = testCorrectCount + tmpACC;
    testErrCount = testErrCount + tmpERR;
end

%get the ACC rate
trainACC = trainCorrectCount / (trainCorrectCount + trainErrCount)
testACC = testCorrectCount / (testCorrectCount + testErrCount)

