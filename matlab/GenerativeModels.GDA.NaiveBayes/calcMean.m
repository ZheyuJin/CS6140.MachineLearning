function [ meanYes, meanNo ] = calcMean( data, label )
%CALCMEAN Seperately calculate mean value for classes label = {0,1}.
%   Detailed explanation goes here
yesRowsIdx = (label(:, end) == 1); % filter rows in label where val == 1
noRowsIdx = (label(:, end) == 0); % filter rows in label where val == 0

yesRows = data(yesRowsIdx,:);
noRows = data(noRowsIdx,:);

meanYes = mean(yesRows);
meanNo = mean(noRows);
end

