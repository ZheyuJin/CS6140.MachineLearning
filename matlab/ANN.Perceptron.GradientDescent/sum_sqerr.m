function [ val ] = sum_sqerr( output, target)
%sum of square error.
val = 0.5 * sum((sum((output -target).^2)));
end

