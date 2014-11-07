function [ outlabel ] = stump_predict( vec, threash )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
outlabel = vec >= threash; % get 0 or 1
outlabel = 2*(outlabel -0.5);  % get -1 or 1
end

