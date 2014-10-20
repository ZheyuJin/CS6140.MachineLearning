function [ prediction, rawOut ] = predictNaiveBayes( priorPos, priorNeg, meanOverall,upperRatioPos, upperRatioNeg, X)
%PREDICTNAIVEBAYES return label decided by naive bayes.
%   In bayes rule, only compoute the numeroator since denominator are the
%   same.
rawOut = zeros(size(X,1),1);
for row = 1: size(X,1)
    dp = X(row, :) ;
    dp = dp - meanOverall;
    
    posProbVec = zeros(1,length(dp));
    negProbVec = zeros(1,length(dp));
    % construct posProbVec for multiplication.
    for col = 1: length(dp)
        if dp(col) > 0
            posProbVec(col) =  upperRatioPos(col)  ;
            negProbVec(col) =  upperRatioNeg(col)  ;
        else
            posProbVec(col) =  1- upperRatioPos(col)  ;
            negProbVec(col) =  1- upperRatioNeg(col)  ;
        end
    end
    
    % rather use log, turn prod into sum; for better numerical accuaracy.
    posNumeratorLog = log(priorPos) + sum(log(posProbVec));
    negNumeratorLog = log(priorNeg) + sum(log(negProbVec));
    
    % since denominator is the same, compare numerator
    rawOut(row) =  posNumeratorLog -negNumeratorLog ;
end
 prediction =  rawOut  >= 0;
end

