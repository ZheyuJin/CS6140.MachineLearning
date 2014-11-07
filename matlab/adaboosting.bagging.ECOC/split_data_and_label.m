function [ data, label] = split_data_and_label( X )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
data = X(:,1:end-1);
label = X(:,end);

end

