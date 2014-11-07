%% initial data loading for random picking adaboost
D = load('e:\\study\\ml\\spam.txt');
D = D(randperm(size(D,1)),:); % shuffle. yes.
%set label as -1/1
D(:,end) = (D(:,end) -0.5) *2;

totalRows= size(D,1);


%separate test set and train set.
siries = (1:size(D,1));
trainSetRow = siries  <= (totalRows *0.9);
testSetRow =  ~trainSetRow ;
trainSet = D(trainSetRow,:);
% size of 2 paercent chunk in test data..
CHUNK_PERCENTAGE = 0.05;
CHUNK_SIZE = round(size(trainSet ,1) * CHUNK_PERCENTAGE );

testSet = D(testSetRow ,:);
%test data and test label
[testData ,testLabel]=split_data_and_label(testSet);
%initials
localTrainSet = [];
siries = (1:size(trainSet,1));
initialTrainRows = siries  <= (length (siries)*CHUNK_PERCENTAGE); % 5 percent  initial data
% move data from train set to local trainset.
localTrainSet= [localTrainSet ;trainSet(initialTrainRows ,:)];
trainSet = trainSet(~initialTrainRows ,:);


testErrContainer = [];
ADA_MAX_ROUND = 10;
%% loop for random pick.
percentage =CHUNK_PERCENTAGE;
while true
    %make train data and train label.
    [trainData, trainLabel] = split_data_and_label(localTrainSet);
    [ alphaVec,stumpVec, localErrtVec, ~, testErrVec,~, ~, ~] = ...
        adaboosing_core( ...
        trainData,trainLabel,testData ,testLabel,ADA_MAX_ROUND ,@getOptStump_fast);
    
    testerr = testErrVec(end)
    testErrContainer = [testErrContainer ;testerr];
    
    [~ , rawOutput ] = calcBoosting(trainSet, alphaVec, stumpVec);
    %tmpTrainSet = [trainSet,rawOutput];
       
    index = randperm(length(rawOutput));
    pickindex = zeros(size(index)); % all zero.
    %set pick index as 1 for top 2percent.
    for i = 1: CHUNK_SIZE
        pickindex(index(i))=1;
    end
    pickindex = logical(pickindex);
    
    localTrainSet = [localTrainSet; trainSet(pickindex,: )];% grow size.
    trainSet = trainSet(~pickindex,: ); % decreasing size.
    
    %     update percentage and check.
    percentage  = percentage +5;
    if  percentage > 50
        break;
    end
end

plot(1:length(testErrContainer),testErrContainer');
randError = testErrContainer;