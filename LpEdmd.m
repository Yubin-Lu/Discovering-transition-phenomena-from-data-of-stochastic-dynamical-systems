%% n2 initial value
clc;clear;
tic
randn('state',200);
rand('state',200);
%% Parameters
alpha = 1.;  %alpha-stable Levy processes
h = 0.0001;  %迭代步长
n1 = 101;  %迭代次数
n2 = 1000000; %初值个数 1e6 sample for additive noise, 3e6 sample for multiplicative noise
% sigma = ones(n1-1,n2);  %%variation
p = 5;  %% order
c = 1;  %% jump bounded
% y = nonlocal(2,1,alpha)
%% Input
X = -2:(4/(n2-1)):2;  %grid point
Z = zeros(n1,n2);
Z(1,:) = X;  %%initial
W1 = sqrt(h)*randn(n1-1,n2);
W2 = h^(1/alpha)*LP(alpha,n1-1,n2,h,c);
% W = h^(1/alpha) * stblrnd(alpha,0,0,1,[n1-1,n2]);
% save Levy.mat W1 W2 
% load('Levy')
for i = 1:n2
    for j = 1:n1-1
%         Z(j+1,i) = Z(j,i) + (1+sin(Z(j,i)))*h + 1*W(j,i);  %% dirft = sin(x)
%         Z(j+1,i) = Z(j,i) + (Z(j,i))*h + 1*W(j,i); %% OU process
%         Z(j+1,i) = Z(j,i) + (Z(j,i) - Z(j,i)^3)*h + 0*W1(j,i)+1*W2(j,i);  %% Euler method for additive noise
        Z(j+1,i) = Z(j,i) + 1*(4*Z(j,i) - Z(j,i)^3)*h + 1*(Z(j,i))*W1(j,i) + 1*W2(j,i);  %% Euler method for multiplicative noise
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
L2 = ((K*B2')-B2')/0.01 - 2*L1_temp;  %% Generator for x^2
L2(find(abs(L2)<0.3)) = 0;
L1
L2

%% Comparison 
% MET
x1 = 1;
x2 = 2;
hh = 0.01;
U = LpMET(x1,x2,hh,c,L2(1)*pi/2,alpha,1,L1,L2);  %% True
u = LpMET(x1,x2,hh,c,L2(1)*pi/2,alpha,0,L1,L2); %% Learning
figure
plot(x1:hh:x2,U,'r','linewidth',1.5)
hold on 
plot(x1:hh:x2,u,'b','linewidth',1.5)
axis([x1 x2 0 0.3])
xlabel('x')
ylabel('T')
title('Mean Exit Time')
legend('true','learn')

% EP
P = LpEP(x1,x2,hh,c,L2(1)*pi/2,alpha,1,L1,L2);  %% True
p = LpEP(x1,x2,hh,c,L2(1)*pi/2,alpha,0,L1,L2); %% Learning
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
    b = 4*x-x.^3;
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
    L(1) = 0;
    n = length(x);
    sigma = zeros(n,1);
    x = x';
    for i = 1:length(L)
        sigma = sigma + L(i)*x.^(i-1);
    end
    sigma = sigma';
end

%% 1-D Levy process(symmetry,alpha-stable)
function y = LP(alpha,m,n,dt,c)
    V = pi/2 * (2*rand(m,n) - 1); 
    W = exprnd(1,m,n);          
%     y = sin(alpha * V) ./ ( cos(V).^(1/alpha) ) .* ...
%         ( cos( V.*(1-alpha) ) ./ W ).^( (1-alpha)/alpha ); 
    y = tan( pi/2 * (2*rand(m,n) - 1) );
    
    while(true)
        ytemp = dt^(1/alpha)*y;
        ind = find(abs(ytemp)>c);
        if isempty(ind)
            break
        end
        l = length(ind);
        VV = pi/2 * (2*rand(l,1) - 1); 
        WW = exprnd(1,l,1);          
%         y(ind) = sin(alpha * VV) ./ ( cos(VV).^(1/alpha) ) .* ...
%         ( cos( VV.*(1-alpha) ) ./ WW ).^( (1-alpha)/alpha ); 
    y(ind) = tan( pi/2 * (2*rand(l,1) - 1) );
    end
    
end

%% Nonlocal term
function y = nonlocal(p,C,alpha)
    eps = 0.001;
%     f = @(x)x^p/abs(x)^(1+alpha); 
    K = (alpha/(2^(1-alpha)*sqrt(pi))) * (gamma((1+alpha)/2)/gamma(1-alpha/2));
    y = quadl(@(x)K*x.^p./(abs(x)).^(1+alpha),-C,-eps) + quadl(@(x)K*x.^p./(abs(x)).^(1+alpha),eps,C);
end




%% Levy process and MET
function u = LpMET(x1,x2,h,c,eps,alpha,bool,L1,L2)
%%-----------------------------------------------------------%%
% u(x)= 0 in R\(a,b)
% h is the step size
% eps is the noise intensity of pure jump Levy process
% alpha is a stable index of Levy process
% bool = 1 underlying system otherwise discovering from data
% L1,L2 are coefficients of drift term and diffusion term
%%----------------------------------------------------------%%
    x = x1:h:x2;  % grid
    n = length(x);
    u = zeros(n,1);
    K = (alpha/(2^(1-alpha)*sqrt(pi))) * (gamma((1+alpha)/2)/gamma(1-alpha/2)); %% Jump measure constant
    C = eps*K*h; %% coefficient of jump integral
    if bool == 1  % underlying system
        b = driftT(x(2:end-1));
        sigma = diffusionT(x(2:end-1));
    else  % learn from data
        b = driftL(x(2:end-1),L1);
        sigma = diffusionL(x(2:end-1),L2);
    end
    
    %center difference
    s1 = 0.5*b/h + 0.5*sigma/(h^2);
    s2 = sigma/(h^2);
    s3 = -0.5*b/h + 0.5*sigma/(h^2);
    A=diag(-s2)+diag(s1(1:end-1),1)+diag(s3(2:end),-1);  %% differential matrix
    D1 = zeros(n,n);
    D2 = D1;  %% integral matrix
    for j = 2:n-1
        y = ((1-j)* h):h:((x2-x1)+(1-j)*h);
        for i = 1:n
            if abs(y(i)) >= 0.01 && abs(y(i)) <= c 
                D1(j,i) = C/(abs(y(i))^(1+alpha));
                D2(j,j) = D2(j,j) + C/(abs(y(i))^(1+alpha));
            else
                D1(j,i) = 0;
            end
        end
    end
    D1(:,n) = 0;
    D = A + D1(2:n-1,2:n-1) - D2(2:n-1,2:n-1);
    yy = -ones(n-2,1);
    u(2:n-1) = D \ yy; 
end

function p = LpEP(x1,x2,h,c,eps,alpha,bool,L1,L2)  %%b and sigma is a vector of the evulating at grid point of the dirft and diffusion term 
%%-----------------------------------------------------------%%
% u(x)= 0 in R\(a,b)
% h is the step size
% eps is the noise intensity of pure jump Levy process
% alpha is a stable index of Levy process
% bool = 1 underlying system otherwise discovering from data
% L1,L2 are coefficients of drift term and diffusion term
%%----------------------------------------------------------%%    
    x = x1:h:x2;  % grid
    x3 = 3;
    n = length(x);
    p = zeros(n,1);
    p(end) = 1;
    K = (alpha/(2^(1-alpha)*sqrt(pi))) * (gamma((1+alpha)/2)/gamma(1-alpha/2)); %% Jump measure constant
    C = eps*K*h; %% coefficient of jump integral
    if bool == 1  % underlying system
        b = driftT(x(2:end-1));
        sigma = diffusionT(x(2:end-1));
    else  % learn from data
        b = driftL(x(2:end-1),L1);
        sigma = diffusionL(x(2:end-1),L2);
    end
    
    %center difference
    s1 = 0.5*b/h + 0.5*sigma/(h^2);
    s2 = sigma/(h^2);
    s3 = -0.5*b/h + 0.5*sigma/(h^2);
    A=diag(-s2)+diag(s1(1:end-1), 1)+diag(s3(2:end),-1);
    D1 = zeros(n,n);
    D2 = D1;  %% integral matrix
    d = zeros(n-2,1);
    for j = 2:n-1
        y = ((1-j)* h):h:((x2-x1)+(1-j)*h);
        ytemp = ((x2-x1)+(1+1-j)*h):h:((x3-x1)+(1-j)*h);
        for i = 1:n
            if i< n && abs(ytemp(i)) >= 0.01 && abs(ytemp(i)) <= c
                d(j-1) = d(j-1) + C/(abs(ytemp(i))^(1+alpha));
            else
                d(j-1) = 0;
            end
            
            if abs(y(i)) >= 0.01 && abs(y(i)) <= c 
                D1(j,i) = C/(abs(y(i))^(1+alpha));
                D2(j,j) = D2(j,j) + C/(abs(y(i))^(1+alpha));
            else
                D1(j,i) = 0;
            end
        end
    end
    D1(:,n) = 0;
    D = A + D1(2:n-1,2:n-1) - D2(2:n-1,2:n-1);
    yy = zeros(n-2,1);
    yy(end) = -s1(end);
    yy = yy - D1(2:n-1,n) - d;
    p(2:n-1) = D \ yy;
end

%%
% w = zeros(101,1);
% for i = 1:100
%     w(i+1) = w(i) + W(i,155431);
% end
% plot(0:0.01:1,w)
% x = -3:0.01:3;
% V = -0.5*x.^2+x.^4;
% plot(x,V,'r')
% hold on
% U = -0.5*x.^2+0.25*x.^4;
% plot(x,U,'b')
% W = -2*x.^2+0.25*x.^4;
% plot(x,W,'g')

