WW = ones(size(D,1),1)/size(D,1);
tic
getOptStump_fast(WW,D(:,1:end-1),D(:,end));
toc

tic
getOptStump(WW,D(:,1:end-1),D(:,end));
toc