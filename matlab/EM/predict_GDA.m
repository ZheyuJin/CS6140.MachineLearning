function [ prediction, rawPrediction] = predict_GDA( priorYes, priorNo, meanYesVec, meanNoVec, std_dev, dataPoints)
%PREDICT_GDA get prediction against given dataPoints by GDA
%   rocData is juse one col of vec. 100*1
%determ = det(covar);


rawPrediction  = zeros(size(dataPoints,1),1);% prealloc
yesProbs = zeros(1, 57);
noProbs = zeros(1, 57);

for rowIdx = 1: size(dataPoints,1)
    line = dataPoints(rowIdx,:);
    for col = 1: 57
    yesProbs(col) = normpdf(line(col), meanYesVec(col), std_dev(col));
    noProbs(col) = normpdf(line(col), meanNoVec(col), std_dev(col));
    end
    %%only exponents needed to compare two values in gaussian prob density
    %%func.
    
    posProb = priorYes * prod(yesProbs);
    negProb = priorNo * prod(noProbs);
    
    
    rawPrediction(rowIdx)  = posProb - negProb ; % think about the -1/2 in the formula.
end
prediction = rawPrediction >=0;
end

