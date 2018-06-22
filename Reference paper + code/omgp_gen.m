%
% [x, Y] = omgp_gen(loghyper, n, D, m)
%
% Generate n output data points for m GPs. Each data point is D
% dimensional. The inputs are unidimensional.
%
% loghyper collects the process hyperparameters [log(timescale); 0.5*log(signalPower); 0.5*log(noisePower)]

function [x, Y] = omgp_gen(loghyper, n, D, m)

rs = sum(100*clock);
randn('state',rs); rand('state',rs);

covfunc = {'covSum', {'covSEiso','covNoise'}};

x = zeros(n*m,1);Y = zeros(n*m,D);
for k = 1:m
    x((k-1)*n+1:k*n) = rand(n,1)*(n-1)+1;
    Y((k-1)*n+1:k*n,:) = chol(feval(covfunc{:}, loghyper, x((k-1)*n+1:k*n)))'*randn(n, D);        % Cholesky decomp.
end

[x, order] = sort(x);
Y = Y(order,:);