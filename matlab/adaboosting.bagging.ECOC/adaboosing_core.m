function [ alphaVec,stumpVec, localErrtVec, trainErrVec,...
    testErrVec,aucVec, W, rawoutput] ...
    = adaboosing_core...
    ( trainData,trainLabel,testData ,testLabel,...
    ROUND_MAX,getStumpFunc)
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
    %     assert(abs(errt_check - errt) < 0.0001);
    %     if(abs(errt-1) < 0.001) % TO FIX NUMERICAL PROBLEMS..
    %         errt =1 - 1e-2;
    %     end
    
    %% get alpha
    alph = log(sqrt(1/errt -1));
    
    assert (alph ~= Inf );
    assert (alph ~= -Inf);
    assert (isreal(alph) );
    
    % store alpha and stump info
    alphaVec(round) = alph;
    stumpVec(round) = stump;
    
    
    %% record local round err ; train & test err; AUC
    localErrtVec (round) = errt;
    [output, rawoutput] = calcBoosting(trainData, alphaVec, stumpVec);
    trianerr = calcErrRate(output, trainLabel);
    trainErrVec (round) = trianerr;
    
    
    [output rawoutput]= calcBoosting(testData,  alphaVec, stumpVec);
    %num_class = length(unique(rawoutput));
    
    testerr = calcErrRate(output, testLabel);
    testErrVec (round) = testerr ;
    
    %% tmpoarary comment out
    %[a,b,c,auc] = perfcurve(testLabel',rawoutput',1);
    auc =0;
    aucVec(round) = auc;
    
    %% adjust weight vect
    predict = stump_predict(trainData(:,stump.featID), stump.threash);
    expo_vec = exp(-alph * (trainLabel .* predict));
    W = W .* expo_vec ;
    W = W / sum(W);% make it  a probablility distribution.
end

end

