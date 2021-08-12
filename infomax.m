clc
clear all
close all
fontsize = 5;
% 0.49281   0.19701
% 0.34844   0.42216
M = 2;
[s1, fs] = audioread('../Data/wav/chase.wav');
[s2, fs] = audioread('../Data/wav/Trumphet.wav');
N = min(length(s1),length(s2));

s1=s1(1:N)/std(s1(1:N));
s2=s2(1:N)/std(s2(1:N));

s=[s1,s2]';

A=rand(M,M);

% Plot histogram of each source signal -
% this approximates pdf of each source.
figure(1);subplot(121),hist(s(1,:),50); drawnow;title('S1 distribution')
set(gca,'FontSize',fontsize)
figure(1);subplot(122),hist(s(2,:),50); drawnow;title('S2 distribution')
set(gca,'FontSize',fontsize)
saveas(gcf,'SourceHistogram.png')
x = A*s; % mixture
W = eye(M,M);
y = W*x;

r=corrcoef([y; s]');
rinitial=abs(r(M+1:2*M,1:M))

maxiter=150; 
eta=0.25; 

hs=zeros(maxiter,1);
gs=zeros(maxiter,1);

% Begin gradient ascent on h 
for iter=1:maxiter
	y = W*x; 
	Y = tanh(y);
	detW = abs(det(W));
	h = ( (1/N)*sum(sum(Y)) + 0.5*log(detW) );
	g = inv(W') - (2/N)*Y*x';
	W = W + eta*g;
	hs(iter)=h; gs(iter)=norm(g(:));
end

figure(2),
subplot(121)
set(gca,'FontSize',fontsize)
plot(hs,'LineWidth',2);title('Function values - Entropy');
xlabel('Iteration');ylabel('h(Y)');
subplot(122)
set(gca,'FontSize',fontsize)
plot(gs,'LineWidth',2);title('Magnitude of Entropy Gradient');
xlabel('Iteration');ylabel('Gradient Magnitude');
saveas(gcf,'Convergence.png')
figure(3)
subplot(321),set(gca,'FontSize',fontsize),plot(s(1,:),'LineWidth',2),ylabel('S1')
subplot(322),set(gca,'FontSize',fontsize),plot(s(2,:),'LineWidth',2),ylabel('S2')
subplot(323),set(gca,'FontSize',fontsize),plot(x(1,:),'LineWidth',2),ylabel('X1')
subplot(324),set(gca,'FontSize',fontsize),plot(x(2,:),'LineWidth',2),ylabel('X2')
subplot(325),set(gca,'FontSize',fontsize),plot(y(1,:),'LineWidth',2),ylabel('Y1')
subplot(326),set(gca,'FontSize',fontsize),plot(y(2,:),'LineWidth',2),ylabel('Y2')
saveas(gcf,'Signals.png')

r=corrcoef([y; s]');
rfinal=abs(r(M+1:2*M,1:M))


