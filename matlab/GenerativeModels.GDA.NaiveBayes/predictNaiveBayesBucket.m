function [ prediction, rawOut] = predictNaiveBayesBucket( priorPos, priorNeg, bucketVec, posProbVec,  negProbVec, data )
%UNTITLED pridict label by naive bayes provided buckets and probs

%% only calcualte numerator.

% bucketPosVec and bucketNegVec are one value per data point
posVec = getBucketProbVec(bucketVec, posProbVec,data);
negVec  = getBucketProbVec(bucketVec, negProbVec,data);
% for numerical accuracy.
posProbLogVec = log(priorPos) + sum(log(posVec'));
negProbLogVec = log(priorNeg) + sum(log(negVec'));
% in two vectors, only compare the log of their sum 
rawOut = posProbLogVec - negProbLogVec;
rawOut = rawOut';

prediction = rawOut>=0;
end

