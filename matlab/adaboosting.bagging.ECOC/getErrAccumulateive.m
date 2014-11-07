function [ errt ] = getErrAccumulateive( W, threash ,X,Y)
%UNTITLED5 Summary of this function goes here
%   X is one feature colum in the traning data.

outlabel= stump_predict(X , threash); % 0 or 1
err_rows = outlabel ~= Y; % get wrong ones.

% sum up items in W for wrong predictions.
errt = sum(W(err_rows,:));
end

