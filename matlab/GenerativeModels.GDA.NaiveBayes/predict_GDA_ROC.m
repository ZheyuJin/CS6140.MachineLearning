function [ rocData ] = predict_GDA_ROC( priorYes, priorNo, meanYes, meanNo, covar, dataPoints,label)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

cov_inv = inv(covar);

rawPrediction  = zeros(size(dataPoints,1),1);% prealloc
posLogVec = zeros(size(dataPoints,1),1);
negLogVec =zeros(size(dataPoints,1),1);

for rowIdx = 1: size(dataPoints,1)
    line = dataPoints(rowIdx,:);
    
    %%only exponents needed to compare two values in gaussian prob density
    %%func.
    expoPos = (line - meanYes) * cov_inv * (line - meanYes)';
    expoNeg =  (line - meanNo) * cov_inv * (line - meanNo)';
    posLogVec(rowIdx) = expoPos;
    negLogVec(rowIdx)  = expoNeg;
    rawPrediction(rowIdx)  = expoNeg - expoPos ; % think about the -1/2 in the formula.
end

plotroc( label', rawPrediction');

%% only make 100 data points for ROC curve; if shallROC is true

ROC_COUNT =100;
rocData = zeros(ROC_COUNT , 2);
diffLogVec = negLogVec - posLogVec;
maxDiff = max(diffLogVec);
minDiff = min(diffLogVec);
diffVec = ((1:ROC_COUNT)/(ROC_COUNT+1) )* (maxDiff-minDiff) + minDiff;

for idx = 1: ROC_COUNT
    diff = diffVec(idx);
    prediction = rawPrediction >= diff ;
    zzz=sum(prediction)
    line = getReportLine(prediction, label);
    rocData(idx,:) = [line(2), line(1)];
end

end

