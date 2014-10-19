function [ equalRate ] = eq_rate( labelX, labelY )
%EQ_RATE claculate the equal rate. 
%   Detailed explanation goes here
logicalVec = labelX == labelY;

equalRate = sum(logicalVec) / length(logicalVec);
end

