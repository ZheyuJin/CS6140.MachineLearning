%initial data loading
D = load('e:\\study\\ml\\spam.txt');
D = D(randperm(size(D,1)),:); % shuffle. yes.

% normalize
%D = normalize(D);

[totalRows,totalCols] = size(D);

kfold = 10;
blockRowCount = int32(totalRows /kfold);

traningACC = zeros(1, kfold);
testACC = zeros(1, kfold);

%%start of k fold 
for curFold  = 0 : kfold -1
    %%data loading for train and test
    testData = D(1+blockRowCount*curFold: blockRowCount*(curFold +1) , :);    
    trainData = [D(1: blockRowCount*curFold ,:);
        D(blockRowCount*(curFold +1)+1: totalRows, : )];
    
    %% make train and test data; and their labels
    trainLabel = trainData(:,end); % label
    trainData = trainData(:,1:end -1); % discard lables
              
    testLabel = testData(:,end); % label    
    testData = testData(:,1:end -1); % discard lables
    %% calc prior prob.
    [priorYes, priorNo] = calcPriors(trainLabel);
   
    %% calc mean for two classes in tranning data.
    [meanYes, meanNo] = calcMean(trainData, trainLabel);
    
    %% find cov for trannig data.
    covar = cov(trainData);
    
    %% find eq_rate for tranning and test data.
    trainOut = predict_GDA(priorYes, priorNo, meanYes, meanNo, covar, trainData);
    traningACC(curFold +1) = eq_rate(trainOut, trainLabel); %index starts from 1
    
    testOut = predict_GDA(priorYes, priorNo, meanYes, meanNo, covar, testData);
    testACC(curFold +1) = eq_rate(testOut, testLabel);%index starts from 1
end
%%report performance by taking mean.
acc_train = mean(traningACC)
acc_test = mean(testACC)

