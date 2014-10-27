function [ bucketVec, posProbVec, negProbVec ] = make9SigmaBucket( data, label )
%make9SigmaBucket make 9 sigma buckets.

%% make bucket vector
% both are 1* 57 vec
meanVec = mean(data);
stdVarVec = std(data);

bucketVec = zeros(10, size(data,2));
% make 10 boundareis
factor = (-4:5 ) -0.5; % will be multiplied by stdvec.

% make 9 sigma bins 
rows = length(factor);
for  row = 1: rows
    bucketVec (row,:) = meanVec + factor(row) * stdVarVec;
end

bucketVec(1,:) = -inf;
bucketVec(end,:) = inf;

%% make prob vector for pos and neg class. 
posData = data(label == 1,:);
negData = data(label == 0,:);

%% make probVec
posProbVec = getProbLaplace(bucketVec, posData);
negProbVec = getProbLaplace(bucketVec, negData);
end

