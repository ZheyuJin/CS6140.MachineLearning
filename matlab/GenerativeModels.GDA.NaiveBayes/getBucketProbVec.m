function [ dataProbVec ] = getBucketProbVec( bucketVec, bucketProbVec,data )
%getBucketProbVec get one probability for each cell of data matrix.
%   Detailed explanation goes here
dataProbVec= zeros(size(data));
[rows,cols] = size(dataProbVec);

for col = 1:cols
    buckets = bucketVec(:,col);
    probs = bucketProbVec(:,col);
    for row = 1: rows
        buckID = findBucketID( buckets , data(row,col) );
        dataProbVec(row,col) = probs(buckID);
    end
end
end

