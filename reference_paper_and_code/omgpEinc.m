% Update elements 1:n in qZ at iteration n, then 100 iterations more
% updating all the elements in qZ

function  [loghyper, convergence] = omgpEinc(loghyper, covfunc, M, X, Y)

% --- Initialize
%[N, D] = size(X);
[N, oD] = size(Y);

maxit = N+100;

logqZ = [zeros(N,1) reshape(loghyper(end-N*(M-1)+1:end),N,M-1)];
logqZ = logqZ - max(logqZ,[],2)*ones(1,M);
logqZ = logqZ-log(sum(exp(logqZ),2))*ones(1,M);
qZ=exp(logqZ);
sn2 = ones(N,1)*exp(2*loghyper(end-N*(M-1)-M+1:end-N*(M-1)))';
logpZ = [0; loghyper(end-N*(M-1)-2*M+2:end-N*(M-1)-M)]'; 
logpZ = logpZ - max(logpZ);
logpZ = logpZ-log(sum(exp(logpZ)));
logpZ = ones(N,1)*logpZ;
convergence = zeros(N*maxit,1);

oldFant = inf;
for iter=1:maxit
    sqB = sqrt(qZ./sn2);
    
    % --- Contribution of independent (modified) GPs
    oldF = 0; hypstart=1;
    a = zeros(N,M);
    cm = covfunc{1}; 
    numhyp = eval(feval(cm{:})); 
    K = feval(cm{:}, loghyper(1:numhyp), X);
    for m = 1:M
        if length(covfunc) > 1
            cm = covfunc{m};numhyp = eval(feval(cm{:}));
            K = feval(cm{:}, loghyper(hypstart:hypstart+numhyp-1), X);
        else
            hypstart = 1;
        end
        R = chol(eye(N) + K.*(sqB(:,m)*sqB(:,m)'));
        sqBY =  (sqB(:,m)*ones(1,oD)).*Y;
        v = R'\sqBY;
        oldF = oldF + 0.5*sum(sum(v.^2)) + oD*sum(log(diag(R)));
        U = R'\diag(sqB(:,m));
        alpha = U'*v;
        
        diagSigma=diag(K)-sum((U*K).^2,1)'; % diag( K - K*U'*U*K )
        mu = K*alpha;
        a(:,m) = -0.5./sn2(:,m).*(sum((Y - mu).^2,2) + oD*diagSigma);
        
        hypstart = hypstart + numhyp;
    end
    if hypstart+2*M+N*(M-1)-2 ~= length(loghyper) error('Incorrect number of parameters');end
    KLZ = sum(sum(qZ.*(logqZ-logpZ),2)); % KL Divergence from the posterior to the prior on Z
    oldF = oldF + oD/2*sum(sum(qZ.*log(2*pi*sn2))) + KLZ;
    convergence(iter) = oldF;

    temp = a +logpZ - oD/2*log(2*pi*sn2);
    logqZ(1:min(iter+M,N),:) = temp(1:min(iter+M,N),:);
    logqZ = logqZ - max(logqZ,[],2)*ones(1,M);
    logqZ = logqZ-log(sum(exp(logqZ),2))*ones(1,M);
    qZ=exp(logqZ);
    
    if iter+M>N && abs(oldF-oldFant)<abs(oldFant)*1e-6
        break
    end
    oldFant = oldF;
end
convergence = convergence(1:iter);
logqZ=logqZ-logqZ(:,1)*ones(1,M);
logqZ = logqZ(:,2:end);
loghyper(end-N*(M-1)+1:end) = logqZ(:);

% This would also update pZ:
%logpZ = log(sum(qZ,1)+ones(1,M)/M);
%logpZ = logpZ - max(logpZ);logpZ = logpZ-log(sum(exp(logpZ)));logpZ=logpZ-logpZ(1); 
%loghyper(end-N*(M-1)-2*M+2:end-N*(M-1)-M) = logpZ(2:end);
