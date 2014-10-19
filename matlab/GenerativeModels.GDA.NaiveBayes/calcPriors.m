function [ yesProb,noProb ] = calcPriors( label)
%CALCPRIORS assuming yes is 1, no is 0 ; return yesPercentage and no
%percentage. 
%   Detailed explanation goes here
total = length(label);
yesCount = sum(label);
yesProb = yesCount / total;
noProb = 1- yesProb;
end

