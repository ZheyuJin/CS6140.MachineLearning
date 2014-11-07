function [ alphaVec,stumpVec]  = adaboosing_training...
    ( trainData,trainLabel,ROUND_MAX,getStumpFunc)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


%initial weight, evenly distributed.
W = ones(size(trainData,1),1) ./size(trainData,1);


%%          now Boosting
%alphaVec = zeros(1,ROUND_MAX); % coeff array
%errtVec = zeros(1,ROUND_MAX);
%stumpVec =struct('featID',0,'threash',0);  % pair of <featid, threash>; dont know how to prealloc.

for round = 1: ROUND_MAX
    [errt, stump] = getStumpFunc(W, trainData, trainLabel);
    %     [errt_check,stump_check] = getOptStump(W, trainData, trainLabel);
    %         assert(abs(errt_check - errt) < 0.0001);
    
    if(abs(errt-1) < 0.0001) % TO FIX NUMERICAL PROBLEMS..
        errt =1 - 1e-5;
    end
    
    %% get alpha
    alph = 0.5 * log(1/errt -1);
    
    assert (alph ~= Inf );% errt =0
    assert (alph ~= -Inf);% errt =1
    assert (isreal(alph) );
    
    % store alpha and stump info
    alphaVec(round) = alph;
    stumpVec(round) = stump;
    
    
    %% adjust weight vect
    predict = stump_predict(trainData(:,stump.featID), stump.threash);
    expo_vec = exp(-alph * (trainLabel .* predict));
    W = W .* expo_vec ;
    W = W / sum(W);% make it  a probablility distribution.
end

end

