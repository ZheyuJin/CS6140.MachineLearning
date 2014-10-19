%initial data loading
D = load('e:\\study\\ml\\spam.txt');
D = D(randperm(size(D,1)),:); % shuffle. yes.

% normalize
%D = normalize(D);

[totalRows,totalCols] = size(D);

kfold = 10;
blockRowCount = int32(totalRows /kfold);
% could be useless. delete if so.
traningACC = zeros(1, kfold);
testACC = zeros(1, kfold);


%% report table

COL_FALSE_POS =1;
COL_FALSE_NEG =2;
COL_OVERALL_ERR =3;

GDA_report = zeros(kfold, 3); %falsePos, falseNeg, overallErrRate =3


%% start of k fold 
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
    
    %%          GDA model          
    % calc prior prob.
    [priorSpam, priorNonSpam] = calcPriors(trainLabel);
   
    % calc mean for two classes in tranning data.
    [meanSpam, meanNonSpam] = calcMean(trainData, trainLabel);
    
    % find cov for trannig data.
    covar = cov(trainData);
    
    % find err for test data.        
    testOut = predict_GDA(priorSpam, priorNonSpam, meanSpam, meanNonSpam, covar, testData);    
    [a,b,c]= getReportLine(testOut, testLabel);%index starts from 1
    GDA_report(curFold +1, : ) = [a,b,c];    
    
    %%          Bernoulli(Boolean model)
    
    
    
    
    
    %%          4 Bucket Histogram model
    
    
    
    
    
    
    %%          9 Bucket Histogram model
    
    
end
%%report performance by taking mean.
GDA_report

