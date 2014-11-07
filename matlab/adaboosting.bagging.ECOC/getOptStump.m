function [errt, stump ] = getOptStump( W, X , Y)
%UNTITLED3 get optimal stump
%   Detailed explanation goes here
optFeat = 1;
optThreash =0;
objvalue =0;

colcount = size(X,2);
for col = 1: colcount
    curCol = X(:,col);
    % try all possible threashold
    threashVec = sort(unique(curCol)) ;
    
    threashVec = [threashVec(1,:)-1 ; threashVec ; threashVec(end,:)+1];
    for i=1:length(threashVec)
        tmpThreash = threashVec(i);
        err= getErrAccumulateive(W, tmpThreash ,curCol,Y);
        
        % this value should be large; to get so-called "best" stump
        tmpObj = abs(0.5 - err) ;        
        %update record. 
        if tmpObj > objvalue
            objvalue = tmpObj;
            optFeat = col;
            optThreash = tmpThreash;
            errt = err;
        end
    end
end

stump.featID = optFeat;
stump.threash = optThreash;
% errt = getErrAccumulateive(W, optThreash,X(:, optFeat),Y);
end

