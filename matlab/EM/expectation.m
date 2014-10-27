function llh = expectation(X,  PI, MIU, SIGMAS)

n = size(X,2);
MC = size(MIU,1);
logRho = zeros(n,MC);

for m = 1:MC
    logRho(:,m) = loggausspdf(X,MIU(m,:),SIGMAS(:,:,m));
end
logRho = bsxfun(@plus,logRho,log(PI));
T = logsumexp(logRho,2);
llh = sum(T)/n; % loglikelihood
end

