function [ out] = mapper(mapping, label)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
out = zeros(size(label));
for i = 1: length(label)
    out(i) = mapping(label(i)+1);
end
end

