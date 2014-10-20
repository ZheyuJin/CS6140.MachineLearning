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
%falsePos, falseNeg, overallErrRate
GDA_report = zeros(kfold, 3);
Bernoulli_report = zeros(kfold, 3);
Bucket4_report = zeros(kfold, 3);
Bucket9_report = zeros(kfold, 3);

ROC_report =[];

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
    [testOut, rawGDAOut] = predict_GDA(priorSpam, priorNonSpam, meanSpam, meanNonSpam, covar, testData);
    GDA_report(curFold +1,: ) = getReportLine(testOut, testLabel);%index starts from 1
       
        
    
    
    %%          Bernoulli(Boolean model)
    featMeanOverall =  mean(trainData);
    [aboveMeanRatioSpam, aboveMeanRatioNonSpam] = aboveMeanRatioEachClass(featMeanOverall, trainData, trainLabel);
    [testOut, rawBernoulliOut] = predictNaiveBayes(priorSpam, priorNonSpam,featMeanOverall,aboveMeanRatioSpam, aboveMeanRatioNonSpam, testData);
    
    Bernoulli_report(curFold +1, : ) = getReportLine(testOut, testLabel);%index starts from 1
    
    %%          4 Bucket Histogram model
    minVec = min(trainData);
    maxVec = max(trainData);
    [meanPos, meanNeg] = meanForEachClass(trainData,trainLabel);
    % passing a super block: 5 * 57
    superblock = [featMeanOverall;meanPos; meanNeg; minVec; maxVec];
    [bucketVec, posProbVec,  negProbVec] = make4Bucket(superblock , trainData, trainLabel);
    [testOut, raw4BucketOut ] = predictNaiveBayesBucket(priorSpam, priorNonSpam,bucketVec, posProbVec,  negProbVec, testData);
    Bucket4_report(curFold +1, : ) = getReportLine(testOut, testLabel);%index starts from 1
    
    %%          9 Bucket Histogram model
    [bucketVec, posProbVec, negProbVec] = make9SigmaBucket(trainData, trainLabel);
    [testOut, raw9BucketOut ] = predictNaiveBayesBucket(priorSpam, priorNonSpam,bucketVec, posProbVec,  negProbVec, testData);
    Bucket9_report(curFold +1, : ) = getReportLine(testOut, testLabel);%index starts from 1
end
%%report performance by taking mean.


GDA_report = [GDA_report ; mean(GDA_report )]
Bernoulli_report  = [Bernoulli_report ; mean(Bernoulli_report )]
Bucket4_report  = [Bucket4_report ; mean(Bucket4_report )]
Bucket9_report  = [Bucket9_report ; mean(Bucket9_report )]

% rawGDAOut
% rawBernoulliOut
% raw4BucketOut
% raw9BucketOut

[a,b,c,aucGDA] = perfcurve(testLabel',rawGDAOut',1)
[a,b,c,aucBernoulli] = perfcurve(testLabel',rawBernoulliOut',1)
[a,b,c,auc4Buck] = perfcurve(testLabel',raw4BucketOut',1)
[a,b,c,auc9Buck] = perfcurve(testLabel',raw9BucketOut',1)

plotroc([testLabel';testLabel';testLabel';testLabel'], [rawGDAOut';rawBernoulliOut'; raw4BucketOut'; raw9BucketOut'] ,'roc curve');
legend( sprintf('GDA AUC %f', aucGDA) , sprintf('Bernoulli AUC %f', aucBernoulli), sprintf('4Bucket AUC %f',auc4Buck ),sprintf('9Bucket AUC  %f',auc9Buck));