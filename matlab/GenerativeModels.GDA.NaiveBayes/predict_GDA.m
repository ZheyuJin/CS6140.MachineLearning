function [ prediction ] = predict_GDA( priorYes, priorNo, meanYes, meanNo, covar, dataPoints)
%PREDICT_GDA get prediction against given dataPoints by GDA
%   Detailed explanation goes here
%determ = det(covar);
cov_inv = inv(covar);

prediction  = zeros(size(dataPoints,1),1);% prealloc

for rowIdx = 1: size(dataPoints,1)
    line = dataPoints(rowIdx,:);
    
    %%only exponents needed to compare two values.
    expoYes = (line - meanYes) * cov_inv * (line - meanYes)';
    expoNo =  (line - meanNo) * cov_inv * (line - meanNo)';
    prediction(rowIdx)  = expoYes < expoNo; % think about the -1/2 in the formula.
end


end

