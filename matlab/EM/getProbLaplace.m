%%
function [probVec ] = getProbLaplace(bucketVec, data)
count = zeros(size(bucketVec) -[1,0]);
%% fill to buckes and cout statistics for data
[rows, cols] = size(data);
for col = 1: cols
    buckets= bucketVec(:, col);
    for row =1 : rows
        rowidx = findBucketID(buckets, data(row,col));
        count (rowidx,col) = 1 + count (rowidx,col);
    end
end

%% laplace smoothing
count = count +1; % add one to each bucket
probVec = count / sum(count(:, 1)); % divide by total occurance count.
end