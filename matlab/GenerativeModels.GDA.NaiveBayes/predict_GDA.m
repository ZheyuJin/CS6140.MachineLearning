function [ prediction, rawPrediction] = predict_GDA( priorYes, priorNo, meanYes, meanNo, covar, dataPoints)
%PREDICT_GDA get prediction against given dataPoints by GDA
%   rocData is juse one col of vec. 100*1
%determ = det(covar);
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
prediction = rawPrediction >=0;
end

