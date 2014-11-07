function [ err_rate ] = calcErrRate(predict, label)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

incorrect = predict ~= label; % get wrong ones.
err_rate= sum(incorrect) / length(incorrect);
end

