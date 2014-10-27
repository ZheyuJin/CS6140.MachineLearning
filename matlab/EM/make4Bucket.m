function [ bucketVec, posProbVec,  negProbVec] = make4Bucket( block, data, label )
%MAKEBUCKET divide the block into 4 buckets for each feature group by class.
%   Detailed explanation goes here

%%  block is 5 * 57 for this problem.
block = sort(block);
block(1,:) = -inf;
block(end,:) = inf;
bucketVec = block;

%% separate by class
posData= data(label ==1,:);
negData= data(label ==0,:);

%% get probVec
posProbVec = getProbLaplace(bucketVec, posData);
negProbVec = getProbLaplace(bucketVec, negData);
end
