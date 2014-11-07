function [errt, stump ] = getOptStump_fast( W, X , Y)
%UNTITLED3 get optimal stump
%   Detailed explanation goes here
optFeat = 1;
optThreash =0;
objValue =0;
errt =0;

colcount = size(X,2);
for col = 1: colcount
    
    
    
    %sort cur colum and weight ==> tmp weight  .
    curCol = X(:,col);
    [curCol, sortedIndex] = sort(curCol );
    tmpW = W(sortedIndex,:);
    tmpLabel = Y(sortedIndex,:);
    
    errWeightSum = sum(tmpW (tmpLabel < 1,:));%initial error when threash is min -1
    
    % try all possible threashold
    threashVec = sort(unique(curCol)) ;
    threashVec = [threashVec(1,:)-1 ; threashVec ; threashVec(end,:)+1];
    
    %initialize index.
    tid =1;
    vid =1;
    
    while tid <= length(threashVec)
        
        %while and(tid <= length(threashVec), col(vid) < threashVec(tid))
        while  curCol(vid) < threashVec(tid)
            if tmpLabel(vid)==1  %make a point from correct => incorrect. sum the weight.
                errWeightSum = errWeightSum + tmpW(vid);
            else % make incorrect point ==> correct. subtract the weight from sum of err weight.
                errWeightSum = errWeightSum - tmpW(vid);
            end
            vid = vid+1;
            
            if vid > length(curCol)
                break
            end
        end
        
        %update
        tmpobj = abs(errWeightSum - 0.5);
        if(tmpobj > objValue)
            objValue= tmpobj  ;
            optFeat = col;
            optThreash = threashVec(tid);
            errt =errWeightSum;
        end
        
        tid = tid  +1;
    end
end

stump.featID = optFeat;
stump.threash = optThreash;
% errt = getErrAccumulateive(W, optThreash,X(:, optFeat),Y);
end

