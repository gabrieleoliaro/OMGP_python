%
% Computes NMSE and NLPD measures for test data
%

function [NMSE, NLPD] = quality(Y, mu, C, pi0)

[Ntst, D, M] = size(mu);

center = zeros(Ntst,D);
for m = 1:M
    center = center + pi0(m)*mu(:,:,m);
end

NMSE = mean(mean((Y-center).^2)./mean((Y-ones(Ntst,1)*mean(Y)).^2));

p = 0;
for m= 1:M
p = p+pi0(m)*exp(-0.5*(Y - mu(:,:,m)).^2./C(:,:,m)-0.5*log(2*pi*C(:,:,m)));
end
NLPD = -mean(mean(log(p)));

