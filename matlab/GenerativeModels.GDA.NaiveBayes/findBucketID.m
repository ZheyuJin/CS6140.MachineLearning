%%
function [bucketId] = findBucketID(buckets, value)
% takes an colum vector and a value, return which bucket does the value
% fall into.
bucketCount   = length(buckets) -1;
for idx = 1: bucketCount  % bucket is one less than row count
    %count for spam class\
    % for a better performance
    %     if  value <= buckets(idx +1)
    if buckets(idx) < value && value <= buckets(idx +1)        
        bucketId = idx;
        break;
    end
end
end