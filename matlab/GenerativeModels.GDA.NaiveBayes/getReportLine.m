function [ falsePosRate, falseNegRate, overallErrRate] = getReportLine( out, label )
%GETREPORTLINE Summary of this function goes here
%   Detailed explanation goes here

accVec = out == label;
overallErrRate = 1- sum(accVec)/length(label);

realSpamIdx     = label(:, end) == 1;
realNonSpamIdx  = label(:, end) == 0;

outSpam     = out(realSpamIdx, :) ;
outNonSpam  = out(realNonSpamIdx, :) ;

% False positive rate: FPR=FP/(FP+TN)
falsePosRate = sum(outNonSpam)/length(outNonSpam);

%True positive rate (or sensitivity): FNR=FN/(TP+FN)
falseNegRate = 1 - sum(outSpam)/length(outSpam);
end

%                           Condition: Spam        Not Spam
% 
%  Test says “Spam”         True positive   |   False positive
%                           ----------------------------------
%  Test says “Not Spam”     False negative  |    True negative
% True positive rate (or sensitivity): TPR=TP/(TP+FN)
% True negative rate (or specificity): TNR=TN/(FP+TN)