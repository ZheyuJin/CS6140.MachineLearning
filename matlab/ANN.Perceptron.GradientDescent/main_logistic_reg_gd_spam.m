D = load('e:\\study\\ml\\spam.txt');
D = D(randperm(size(D,1)),:); % shuffle. yes.

% normalize
D = normalize(D);

[totalRows,totalCols] = size(D);

kfold = 10;
blockRowCount = int32(totalRows /kfold);

%data for ACC rate
trainErrCount =0;
trainCorrectCount =0;
testErrCount =0;
testCorrectCount =0;

%data for confusion matrix
totalTP =0;
totalTN =0;
totalFP =0;
totalFN =0;


%preallocate train and test matrix
labelTrain =[];
labelTest =[];
outTrain=[];
outTest =[];


for curFold  = 0 : kfold -1
    %data loading for train and test
    testData = D(1+blockRowCount*curFold: blockRowCount*(curFold +1) , :);
    
    trainData = [D(1: blockRowCount*curFold ,:);
        D(blockRowCount*(curFold +1)+1: totalRows, : )];
    
    labelTrain = trainData(:,end); % label
    [train_row, train_col] = size(trainData);
    free = ones(train_row,1); % free terms
    trainData = [free,trainData]; % add free terms
    trainData = trainData(:,1:totalCols ); % discard lables
    
    %initial weight vector w
    w = (rand(train_col,1) - 0.5); %% (-0.5, 0.5)
    w(1) =1; % weight for the free term.
    
    %jump into GD..!! very bad performace. slow to converge...
    w = logistic_ga_batch_update(0.001, w, trainData, labelTrain, 1000);
    
    %train MSE
    outTrain = vec_sigmoid(trainData * w);
    
    
    [truePos,trueNeg, falsePos,falseNeg] = spam_acc_err_count(outTrain , labelTrain,0.5);
    trainCorrectCount = trainCorrectCount + truePos + trueNeg;
    trainErrCount = trainErrCount + falsePos + falseNeg;
    totalTP = totalTP + truePos;
    totalTN = totalTN + trueNeg;
    totalFP =totalFP + falsePos;
    totalFN =totalFN + falseNeg;
    
    %test MSE
    labelTest = testData(:,end); % label
    [test_row, test_col] = size(testData);
    free = ones(test_row,1); % free terms
    testData = [free,testData]; % add free terms
    testData = testData(:,1:totalCols ); % discard lables
    outTest = vec_sigmoid(testData * w);
    
    [truePos,trueNeg, falsePos,falseNeg]  = spam_acc_err_count(outTest , labelTest , 0.5);
    testCorrectCount = testCorrectCount + truePos + trueNeg;
    testErrCount = testErrCount + falsePos + falseNeg;
    totalTP = totalTP + truePos;
    totalTN = totalTN + trueNeg;
    totalFP =totalFP + falsePos;
    totalFN =totalFN + falseNeg;
    
    
end

%get the ACC rate
trainACC = trainCorrectCount / (trainCorrectCount + trainErrCount)
testACC = testCorrectCount / (testCorrectCount + testErrCount)

total = totalTP+totalTN+totalFP+totalFN
tp = totalTP/total
tn = totalTN/total
fp = totalFP/total
fn = totalFN/total

%plotting roc.
totalLabelLogistic = [labelTrain; labelTest];
totalOutLogistic = [outTrain; outTest];
plotroc(totalLabelLogistic',totalOutLogistic');

%compute AUC 
[X,Y,T,LogisticAUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(totalLabelLogistic',totalOutLogistic',1);
LogisticAUC

