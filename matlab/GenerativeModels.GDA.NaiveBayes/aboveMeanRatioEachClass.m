function [ upperRatioPos, upperRatioNeg ] = aboveMeanRatioEachClass( meanOverall, data, label )
%MEANVECEACHCLASS mean vector for positive and negative cases.

%% separate  datapoints for two classes
posIdx = label ==1;
negIdx = label ==0;
assert (sum(posIdx + negIdx)== length(posIdx));
posData = data(posIdx, :);
negData = data(negIdx, :);

%% make indicators for whether feature > mean or not.
% subtract from every line
for row = 1: size(posData,1)      
    posData(row, :) = posData(row, :) - meanOverall;
end

for row = 1: size(negData,1)      
    negData(row, :) = negData(row, :) - meanOverall;
end
% make indicators
posUpperVec = posData > 0;
negUpperVec = negData > 0;

%% laplace smoothing used. 
% upperRatioPosPrev = (sum(posUpperVec) ) / (size(posUpperVec,1 )) ;
% need to make sure the diff value is really small (upperRatioPosPrev - upperRatioPos )'
upperRatioPos = (sum(posUpperVec) +1) / (size(posUpperVec,1 ) +2);
upperRatioNeg = (sum(negUpperVec) +1)/ (size(negUpperVec,1) +2);
end

