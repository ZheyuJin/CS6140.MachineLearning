function [ meanPos, meanNeg ] = meanForEachClass( data,label )
%MEANFOREACHCLASS feature mean value for each class.
%   Detailed explanation goes here
posIdx = label ==1;
negIdx = label ==0;

meanPos = mean(data(posIdx, :));
meanNeg= mean(data(negIdx, :));
end

