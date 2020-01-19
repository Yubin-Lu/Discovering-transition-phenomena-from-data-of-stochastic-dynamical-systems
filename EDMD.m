%% n2 initial value
clc;clear;
tic
randn('state',200);
rand('state',200);
%% Parameters
h = 0.0001;  %迭代步长
n1 = 101;  %迭代次数
n2 = 1000000; %初值个数 1e6 sample for additive noise, 3e6 sample for multiplicative noise
% sigma = ones(n1-1,n2);  %%variation
p = 5;  %%order

%% Input
% X = 4*rand(1,n2) - 2;  %Uniform distribution

% % adapt grid
% X = zeros(1,n2);
% X(1:2*n2/5) = -1.5:(1/(2*n2/5-1)):-0.5;
% X(2*n2/5+1:3*n2/5) = -0.5:(1/(n2/5-1)):0.5;
% X(3*n2/5+1:end) = 0.5:(1/(2*n2/5-1)):1.5;

% % adapt grid
% X = zeros(1,n2);
% X(1:2*n2/5) = -0.75:(0.5/(2*n2/5-1)):-0.25;
% X(2*n2/5+1:3*n2/5) = -0.25:(0.5/(n2/5-1)):0.25;
% X(3*n2/5+1:end) = 0.25:(0.5/(2*n2/5-1)):0.75;

X = -2:(4/(n2-1)):2;  %grid point

Z = zeros(n1,n2);
Z(1,:) = X;  %%initial
W = sqrt(h)*randn(n1-1,n2);
% save BM1D.mat W
% load('BM')
for i = 1:n2
    for j = 1:n1-1
%         Z(j+1,i) = Z(j,i) + (1+sin(Z(j,i)))*h + 1*W(j,i);  %% dirft = sin(x)
%         Z(j+1,i) = Z(j,i) + (Z(j,i))*h + 1*W(j,i); %% OU process
%         Z(j+1,i) = Z(j,i) + (Z(j,i) - Z(j,i)^3)*h + 1*W(j,i);  %% Euler method for additive noise
        Z(j+1,i) = Z(j,i) + (4*Z(j,i) - Z(j,i)^3)*h + (Z(j,i))*W(j,i);  %% Euler method for multiplicative noise
    end
end
Y = Z(end,:);

%% EDMD algrithm
G = zeros(p+1,p+1);
A = zeros(p+1,p+1);
psiX = zeros(n2,p+1);  %% observable for X
psiY = zeros(n2,p+1);  %% observable for Y
for i = 1:n2
    psiX(i,:) = basis(X(i),p);
    psiY(i,:) = basis(Y(i),p);
    G = G + psiX(i,:)' * psiX(i,:);
    A = A + psiX(i,:)' * psiY(i,:);
end
G = G / n2;
A = A / n2;
K = pinv(G) * A;  %%Koopman operator
B1 = zeros(1,p+1);
B2 = zeros(1,p+1);

% Polynomial basis
B1(2) = 1;
B2(3) = 1;
L1 = ((K*B1')-B1')/0.01; %% Generator for x
L1(find(abs(L1)<0.1)) = 0;
L1_temp = [0;L1(1:end-1)];
L2 = ((K*B2')-B2')/0.01 - 2*L1_temp ;  %% Generator for x^2
L2(find(abs(L2)<0.3)) = 0;

% % Hermite polynomial basis
% B1(2) = 1;
% B2(1) = 1;
% B2(3) = 1;
% L1 = ((K*B1')-B1')/0.01; %% Generator for x
% L1(find(abs(L1)<0.1)) = 0;
% L1_temp = [0;L1(1:end-1)];
% L2 = ((K*B2')-B2')/0.01 - 2*L1_temp ;  %% Generator for x^2 
% L2(find(abs(L2)<0.3)) = 0;
L1
L2


%% Comparison 
% MET
x1 = 1;
x2 = 2;
U = MET(1,L1,L2,x1,x2);  %% True
u = MET(0,L1,L2,x1,x2); %% Learning
figure
plot(x1:0.01:x2,U,'r','linewidth',1.5)
hold on 
plot(x1:0.01:x2,u,'b','linewidth',1.5)
xlabel('x')
ylabel('T')
axis([x1 x2 0 0.3])
title('Mean Exit Time')
legend('true','learn')

% EP
P = EP(1,L1,L2,x1,x2);  %% True
p = EP(0,L1,L2,x1,x2); %% Learning
figure
plot(x1:0.01:x2,P,'r','linewidth',1.5)
hold on 
plot(x1:0.01:x2,p,'b','linewidth',1.5)
xlabel('x')
ylabel('P')
title('Escape Probability')
legend('true','learn')
toc

%% basis function
function  psi = basis(x,p)

    psi = zeros(p+1,1);
    % Polynomial basis
    for i = 1:p+1
        psi(i) = x.^(i-1);
    end
%     %Hermite polynomial
%     psi(1) = x.^0;
%     psi(2) = x;
%     for i = 3:p+1
%         psi(i) = x.*psi(i-1) - (i-1)*psi(i-2);
%     end
end

%% Underlying law
%Drift term: True
function b = driftT(x)
    b = 1*(4*x-x.^3);
%     b = sin(x);
end
%Diffusion term: True
function sigma = diffusionT(x)
%     sigma = 1*ones(size(x));  % additive noise
    sigma = x.^2;  % multiplicative noise
end

%% Learning from data
%Drift term: learning from data
function b = driftL(x,L)
    n = length(x);
    b = zeros(n,1);
    x = x';
    for i = 1:length(L)
        b = b + L(i)*x.^(i-1);
    end
    b = b';
end

%Diffusion term: learning from data
function sigma = diffusionL(x,L)
    n = length(x);
    sigma = zeros(n,1);
    x = x';
    for i = 1:length(L)
        sigma = sigma + L(i)*x.^(i-1);
    end
    sigma = sigma';
end

%% Calculating MET
function u = MET(bool,L1,L2,x1,x2)  %%b and sigma is a vector of the evulating at grid point of the dirft and diffusion term 
    h = 0.01;  %step
    x = x1:h:x2;  % grid
%     x = 0:h:1;  % grid
    n = length(x);
    u = zeros(n,1);
    if bool == 1  % underlying system
        b = driftT(x(2:end-1));
        sigma = diffusionT(x(2:end-1));
    else  % learn from data
        b = driftL(x(2:end-1),L1);
        sigma = diffusionL(x(2:end-1),L2);
    end
    
    %center difference
%     s1 = b/h + 0.5*sigma/(h^2);
%     s2 = b/h + sigma/(h^2);
%     s3 = 0.5*sigma/(h^2);
    s1 = 0.5*b/h + 0.5*sigma/(h^2);
    s2 = sigma/(h^2);
    s3 = -0.5*b/h + 0.5*sigma/(h^2);
    A=diag(-s2)+diag(s1(1:end-1), 1)+diag(s3(2:end),-1);
    y = -ones(n-2,1);
    u(2:n-1) = A \ y; 
end
    
%% Calculating EP
function p = EP(bool,L1,L2,x1,x2)  %%b and sigma is a vector of the evulating at grid point of the dirft and diffusion term 
    h = 0.01;  %step
    x = x1:h:x2;  % grid
%     x = 0:h:1;  % grid
    n = length(x);
    p = zeros(n,1);
    p(end) = 1;
    if bool == 1  % underlying system
        b = driftT(x(2:end-1));
        sigma = diffusionT(x(2:end-1));
    else  % learn from data
        b = driftL(x(2:end-1),L1);
        sigma = diffusionL(x(2:end-1),L2);
    end
    
    %center difference
%     s1 = b/h + 0.5*sigma/(h^2);
%     s2 = b/h + sigma/(h^2);
%     s3 = 0.5*sigma/(h^2);
    s1 = 0.5*b/h + 0.5*sigma/(h^2);
    s2 = sigma/(h^2);
    s3 = -0.5*b/h + 0.5*sigma/(h^2);
    A=diag(-s2)+diag(s1(1:end-1), 1)+diag(s3(2:end),-1);
    y = zeros(n-2,1);
    y(end) = -s1(end);
    p(2:n-1) = A \ y;
end
    
    
    


