function [ ] = adaboost( D, ROUND_MAX, getStumpFunc)
%adaboost runs ada boots.
%   D is data set with the last col is the label, must be a binary label(0
%   or 1), other values are not allowd.
%
%   getStumpFunc is the function used to get the

[totalRows,totalCols] = size(D);

kfold = 10;
blockRowCount = int32(totalRows /kfold);

% looks messy, but what it is doing is just spilit the whole dataset into two parts: test and tranning.
curFold =1;

%% data loading for train and test
testData = D(1+blockRowCount*curFold: blockRowCount*(curFold +1) , :);
trainData = [D(1: blockRowCount*curFold ,:);
    D(blockRowCount*(curFold +1)+1: totalRows, : )];

%% make train and test data; and their labels
trainLabel = trainData(:,end); % label
trainLabel = (trainLabel -0.5)*2; % map 0/1 ==> -1/1
trainData = trainData(:,1:end -1); % discard lables

testLabel = testData(:,end); % label
testLabel  = (testLabel -0.5)*2;
testData = testData(:,1:end -1); % discard lables

%% call boosting
[~,~, localErrtVec, trainErrVec, testErrVec,aucVec, ~,rawoutput] = adaboosing_core( trainData,trainLabel,testData ,testLabel,ROUND_MAX,getStumpFunc);
axis = 1: ROUND_MAX;


%% ploting 3 * 1  subplots
figure() ;
subplot(2 ,2 ,1);
plot(axis,localErrtVec);
title('local err');

subplot(2 ,2 ,2);
plot(axis,trainErrVec, axis,testErrVec);
legend('train','test');
title('train and test err');


subplot(2 ,2 ,3) ;
plot(axis,aucVec);
title('AUC');


%% plot ROC
figure() ;
plotroc(testLabel' ==1,rawoutput' );


end

