function [ meanYesVec, meanNoVec ] = calcMean_eachfeature( data, label )
%CALCMEAN Seperately calculate mean value for classes label = {0,1}.
%   Detailed explanation goes here
meanYesVec = zeros(1,57);
meanNoVec = zeros(1,57);
for col = 1: 57
yesRowsIdx = (label(:, end) == 1); % filter rows in label where val == 1
noRowsIdx = (label(:, end) == 0); % filter rows in label where val == 0

yesRows = data(yesRowsIdx,col);
noRows = data(noRowsIdx,col);

meanYesVec(col) = mean(yesRows);
meanNoVec(col) = mean(noRows);
end

