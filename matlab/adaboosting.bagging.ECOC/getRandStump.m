function [errt, stump ] = getRandStump( W, X , Y)
%UNTITLED3 get optimal stump
%   Detailed explanation goes here
feat_rand = randi([1 size(X,2)]); %rand col
threash_rand = X(randi([1 size(X,1)]),feat_rand);%rand row
stump.featID = feat_rand;
stump.threash = threash_rand; 
errt = getErrAccumulateive(W, threash_rand,X(:, feat_rand),Y);
end

