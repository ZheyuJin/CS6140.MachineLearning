%%% all student comment starts with %my.comment%

function [label, model, llh] = emgm(X, init)
% Perform EM algorithm for fitting the Gaussian mixture model.
% X: d x n data matrix
% init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
% Written by Michael Chen (sth4nth@gmail.com).
%% initialization
fprintf('EM for Gaussian mixture: running ... \n');
R = initialization(X,init);

%my.comment%  pick max value in every row; discard first component
%my.comment%  label(1,:) contains the index of max valued items in R's
%every row.
[~,label(1,:)] = max(R,[],2); 
%my.comment%   R is now regrouped by colum, sorted in reversed order, 
%, key is the max value is each colum, 
R = R(:,unique(label));

tol = 1e-10;
maxiter = 500;
llh = -inf(1,maxiter);
converged = false;
t = 1;
while ~converged && t < maxiter
    t = t+1;
    model = maximization(X,R); %my.comment% a M step.
    [R, llh(t)] = expectation(X,model); %my.comment% an E step. 
    
    [~,label(:)] = max(R,[],2);
    u = unique(label); % non-empty components
    if size(R,2) ~= size(u,2)
        R = R(:,u); % remove empty components
    else
        converged = llh(t)-llh(t-1) < tol*abs(llh(t)); %my.comment% converge check. 
    end
    figure(gcf); clf;
    spread(X,label);
    muA = model.mu;
    SigmaA = model.Sigma;
    wA = model.weight;
    k = size(muA,2);
    % figure(12); clf;
    % for i=1:k
    % mu1 =muA(i,:)
    % Sigma1=SigmaA(i,:)
    % w1=wA(i)
    % xx= mvnrnd(mu1, Sigma1, 1000);
    % yy= mvnpdf(xx,mu1,Sigma1);
    % plot3(xx(:,1), xx(:,2), yy, '.b'); hold on;
    % end
    
    pause;
    
    
    
end
llh = llh(2:t);
if converged
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




