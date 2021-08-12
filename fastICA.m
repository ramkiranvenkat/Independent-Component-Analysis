clc
clear all
close all

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
figure(1);subplot(121),hist(s(1,:),50); drawnow;
figure(1);subplot(122),hist(s(2,:),50); drawnow;

x = A*s; % mixture

%centering the data
x = x-mean(x')';
xxt = x*x';
[Ev,l] = eig(xxt);
V = Ev*sqrt(inv(l))*Ev';
z = V*x;

maxitr = 100;

for i=1:M % change
	temp = randn(M,1);
	W(:,i) = temp/norm(temp);
end
W = chol(inv(W*W'))'*W;

for itr = 1:maxitr
	for i=1:M
		w = W(:,i);
		e1 = zeros(M,1);
		e2 = 0;
		onebyN = 1/length(z);
		for j=1:length(z)
			e1 = e1 + onebyN*z(:,i)*g(1,w'*z(:,i));
			e2 = e2 + onebyN*gd(1,w'*z(:,i));
		end
		W(:,i) = real(e1 - e2*w);
	end
	W = chol(inv(W*W'))'*W;
end
y = W'*z;

figure
subplot(321),plot(s(1,:),'LineWidth',2),ylabel('S1')
subplot(322),plot(s(2,:),'LineWidth',2),ylabel('S2')
subplot(323),plot(x(1,:),'LineWidth',2),ylabel('X1')
subplot(324),plot(x(2,:),'LineWidth',2),ylabel('X2')
subplot(325),plot(y(1,:),'LineWidth',2),ylabel('Y1')
subplot(326),plot(y(2,:),'LineWidth',2),ylabel('Y2')
