clc
clear all
close all

num_sources = 3;
num_mixtures = num_sources;
num_samples = 5000;

max_mask_len= 500;
n = 8;

h=2; t = n*h; lambda = 2^(-1/h);temp = [0:t-1]'; lambdas = ones(t,1)*lambda;
mask = lambda.^temp;
mask1 = mask/sum(abs(mask));
h=4; t = n*h; lambda = 2^(-1/h);temp = [0:t-1]'; lambdas = ones(t,1)*lambda;
mask = lambda.^temp;
mask2 = mask/sum(abs(mask));
h=8; t = n*h; lambda = 2^(-1/h);temp = [0:t-1]'; lambdas = ones(t,1)*lambda;
mask = lambda.^temp;
mask3 = mask/sum(abs(mask));

sources = randn(num_samples,num_sources);
sources(:,1)=filter(mask1,1,sources(:,1));
sources(:,2)=filter(mask2,1,sources(:,2));
sources(:,3)=filter(mask3,1,sources(:,3));

sources=sources';

A = randn(num_sources,num_sources)';
mixtures = A*sources;

shf = 1;
lhf = 900000;

h=shf; t = n*h; lambda = 2^(-1/h); temp = [0:t-1]';
lambdas = ones(t,1)*lambda; mask = lambda.^temp;
mask(1) = 0; mask = mask/sum(abs(mask)); mask(1) = -1;
s_mask=mask; s_mask_len = length(s_mask);

h=lhf;t = n*h; t = min(t,max_mask_len); t=max(t,1);
lambda = 2^(-1/h); temp = [0:t-1]';
lambdas = ones(t,1)*lambda; mask = lambda.^temp;
mask(1) = 0; mask = mask/sum(abs(mask)); mask(1) = -1;
l_mask=mask; l_mask_len = length(l_mask);


S=filter(s_mask,1,mixtures')';
L=filter(l_mask,1,mixtures')';

U=cov(S',1);
V=cov(L',1);

cl=V;
cs=U;

w=randn(1,num_sources);
w=w/norm(w);
w0=w;

y0=w0*mixtures;
rs0=corrcoef([y0; sources]');
abs(rs0(1,2:4))

eta=1e-1;
maxiter=100;

gs=zeros(maxiter,1); % gradient magnitude |g|
Fs=zeros(maxiter,1); % function value F

for i=1:maxiter
	% Get value of function F
	Vi = w*cl*w';
	Ui = w*cs*w';
	F = log(Vi/Ui);
	% Get gradient
	g = 2*w*cl./Vi - 2*w*cs./Ui;
	% Update w
	w = w + eta*g;
	% Record results ...
	Fs(i)=F;
	gs(i)=norm(g);
end

figure(1); plot(Fs); xlabel('Iteration Number'); ylabel('F=log(V/U)');
figure(2); plot(gs); xlabel('Iteration Number'); ylabel('Gradient Magnitude');

y1=w*mixtures;
rs=corrcoef([y1; sources]');
abs(rs(1,2:4))

[Wtemp d]=eig(V,U);
W=Wtemp'; W=real(W);
ys = W*mixtures;
a=[sources; ys]'; c=corrcoef(a);
rs=c(1:num_sources,num_sources+1:num_sources*2);
abs(rs)
