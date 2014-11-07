function [ output , rawOutput ] = calcBoosting(X, alphaVec, stumpVec)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
sum =zeros(size(X,1),1);
for col = 1: length(alphaVec)
    alph = alphaVec(col);
    stump = stumpVec(col);
    % wrong to use Y here.
    % sum = sum + alph * (Y .* stump_predict (X(:, stump.featID), stump.threash));
    feat = X(:, stump.featID);
    sum = sum + alph * stump_predict( feat , stump.threash);
end
output = sign(sum);
rawOutput = sum;
end

