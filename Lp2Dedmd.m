%%
clc;clear;
tic
% randn('state',100);
% rand('state',100);
%% Parameters
h = 0.0001;  %迭代步长
n1 = 101;  %迭代次数
n2 = 1000000; %初值个数
p = 3;  %%order
alpha = 1;
sigma = 1.;
Teps1 = 1;
Teps2 = 1;
mu = 3;
omega = 0;
C = 1;
% [int1,int2] = nonlocal(p,C,alpha)
%% Input
% % adapt grid
% x1 = zeros(1,n2);
% x2 = zeros(1,n2);
% x1(1:2*n2/5) = -1.5:(1/(2*n2/5-1)):-0.5;
% x1(2*n2/5+1:3*n2/5) = -0.5:(1/(n2/5-1)):0.5;
% x1(3*n2/5+1:end) = 0.5:(1/(2*n2/5-1)):1.5;
% x2(1:2*n2/5) = -1.5:(1/(2*n2/5-1)):-0.5;
% x2(2*n2/5+1:3*n2/5) = -0.5:(1/(n2/5-1)):0.5;
% x2(3*n2/5+1:end) = 0.5:(1/(2*n2/5-1)):1.5;

% Uniform distribution
x1 = 2*rand(1,n2) - 1;
x2 = 2*rand(1,n2) - 1;

z1 = zeros(n1,n2);
z2 = zeros(n1,n2);
z1(1,:) = x1;  %%initial
z2(1,:) = x2;  %%initial
% randn('state',100);
W1 = sqrt(h)*randn(n1-1,n2);
% randn('state',100);
W2 = sqrt(h)*randn(n1-1,n2);
LP1 = h^(1/alpha)*LP(alpha,n1-1,n2,h,C);
LP2 = h^(1/alpha)*LP(alpha,n1-1,n2,h,C);
for i = 1:n2
    for j = 1:n1-1
        % linear
            % multi
%         z1(j+1,i) = z1(j,i) + 3*z1(j,i)*h + 1*sigma*z1(j,i)*W1(j,i) + 1*LP1(j,i);
%         z2(j+1,i) = z2(j,i) + (2*z1(j,i)+z2(j,i))*h + 1*sigma*z2(j,i)*W2(j,i) + 1*LP2(j,i);
        % nonlinear
            %add
%         z1(j+1,i) = z1(j,i) + (3*z1(j,i)-z2(j,i)^2)*h + 1*sigma*W1(j,i) + 1*LP1(j,i);
%         z2(j+1,i) = z2(j,i) + (2*z1(j,i)+z2(j,i))*h + 1*sigma*W2(j,i) + 1*LP2(j,i);
            % multi
        z1(j+1,i) = z1(j,i) + (3*z1(j,i)-z2(j,i)^2)*h + 1*sigma*z1(j,i)*W1(j,i) + 1*LP1(j,i);
        z2(j+1,i) = z2(j,i) + (2*z1(j,i)+z2(j,i))*h + sigma*z2(j,i)*W2(j,i) + 1*LP2(j,i);
%         z1(j+1,i) = z1(j,i) + (-mu*z1(j,i) - omega*z2(j,i)+z1(j,i)^2*z2(j,i))*h + sigma*W1(j,i);  %% Euler method for additive noise
%         z2(j+1,i) = z2(j,i) + (omega*z1(j,i) - mu*z2(j,i))*h + sigma*W2(j,i);
%         Z(j+1,i) = Z(j,i) + (Z(j,i) - Z(j,i)^3)*h + (Z(j,i))*W(j,i);  %% Euler method for multiplicative noise
    end
end
y1 = z1(end,:);
y2 = z2(end,:);
x = [x1;x2];
y = [y1;y2];

%% EDMD algrithm
P = (p+1)*(p+2)/2;
G = zeros(P);
A = zeros(P);
% psiX = zeros(n,P);  %% observable for X
% psiY = zeros(n,P);  %% observable for Y
psiX = zeros(n2,P);  %% observable for X
psiY = zeros(n2,P);  %% observable for Y
% for i = 1:n
for i = 1:n2
    psiX(i,:) = basis(x(:,i),p);
    psiY(i,:) = basis(y(:,i),p);
%     psiX(i,:) = basis(x(:,i),p);
%     psiY(i,:) = basis(x(:,i+1),p);
    G = G + psiX(i,:)' * psiX(i,:);
    A = A + psiX(i,:)' * psiY(i,:);
end
G = G / n2;
A = A / n2;
% G = G / n;
% A = A / n;
K = pinv(G) * A;  %%Koopman operator
% b1
Bb1 = zeros(1,P);
Bb1(2) = 1;
Lb1 = ((K*Bb1')-Bb1')/(h*(n1-1)); %% Generator for x
% Lb1 = ((K*Bb1')-Bb1')/0.01; %% Generator for x
Lb1(find(abs(Lb1)<0.3)) = 0;
% b2
Bb2 = zeros(1,P);
Bb2(3) = 1;
Lb2 = ((K*Bb2')-Bb2')/(h*(n1-1)); %% Generator for x
% Lb2 = ((K*Bb2')-Bb2')/0.01; %% Generator for x
Lb2(find(abs(Lb2)<0.3)) = 0;
Lb1
Lb2
% sigma
Bs11 = zeros(1,P);
Bs11(4) = 1;
Lb1_temp = [0;Lb1(1);0;Lb1(2);Lb1(3);0;Lb1(4);Lb1(5);Lb1(6);0;];
% Ls11 = ((K*Bs11')-Bs11')/0.01- 2*Lb1_temp;  %% Generator for x^2
% Ls11 = ((K*Bs11')-Bs11')/0.01- 2*[0;0;0;3;0;0;0;0;0;0];  %% Generator for x^2 and linear
Ls11 = ((K*Bs11')-Bs11')/0.01- 2*Lb1_temp;  %% Generator for x^2 and nonlinear
Ls11(find(abs(Ls11)<0.3)) = 0;
Ls11

Bs22 = zeros(1,P);
Bs22(6) = 1;
Lb2_temp = [0;0;Lb2(1);0;Lb2(2);Lb2(3);0;Lb2(4);Lb2(5);Lb2(6)];
% Ls22 = ((K*Bs22')-Bs22')/0.01 - 2*Lb2_temp;  %% Generator for x^2
% Ls22 = ((K*Bs22')-Bs22')/0.01 - 2*[0;0;0;0;2;1;0;0;0;0];  %% Generator for x^2 and linear
Ls22 = ((K*Bs22')-Bs22')/0.01 - 2*Lb2_temp;  %% Generator for x^2 and nonlinear
Ls22(find(abs(Ls22)<0.3)) = 0;
Ls22


%% Comparison 
% MET
xx1 = 1;xx2 = 2;yy1 = 1;yy2 = 2;
TLb1 = [0,3,0,0,0,-1,0,0,0,0];
TLb2 = [0,2,1,0,0,0,0,0,0,0];
TLs11 = [0,0,0,1,0,0,0,0,0,0];
TLs22 = [0,0,0,0,0,1,0,0,0,0];
eps1 = Ls11(1)*pi/2;
eps2 = Ls22(1)*pi/2;
Ls11(1) = 0;
Ls22(1) = 0;
[XX,YY,U] = MET2dEP(1,0,TLb1,TLb2,TLs11,TLs22,xx1,xx2,yy1,yy2,C,Teps1,Teps2,alpha);    %% True
[~,~,u] = MET2dEP(1,0,Lb1,Lb2,Ls11,Ls22,xx1,xx2,yy1,yy2,C,eps1,eps2,alpha);   %% Learning
figure
mesh(XX,YY,U')
xlabel('x')
ylabel('y')
zlabel('u')
title('Mean Exit Time for True')
axis([1 2 1 2 0 0.05])
figure
mesh(XX,YY,u')
xlabel('x')
ylabel('y')
zlabel('u')
title('Mean Exit Time for Learning')
axis([1 2 1 2 0 0.05])

% EP
[~,~,P] = MET2dEP(0,0,TLb1,TLb2,TLs11,TLs22,xx1,xx2,yy1,yy2,C,Teps1,Teps2,alpha);    %% True
[~,~,p] = MET2dEP(0,0,Lb1,Lb2,Ls11,Ls22,xx1,xx2,yy1,yy2,C,eps1,eps2,alpha);   %% Learningfigure
% PP = P(:,end:-1:1);
% pp = p(:,end:-1:1);
% PP = P';
% pp = p';
figure
mesh(XX,YY,P')
xlabel('x')
ylabel('y')
zlabel('P')
title('Escape Probability for True')
figure
mesh(XX,YY,p')
xlabel('x')
ylabel('y')
zlabel('p')
title('Escape Probability for Learning')

toc

%% basis function
function  psi = basis(x,p)
%     [m,n] = size(x);
    k = (p+1)*(p+2)/2;
    psi = zeros(k,1);
    count = 1;
    for i = 1:p+1
        for j = 1:i
            psi(count) = x(1)^(i-j) * x(2)^(j-1);
            count = count + 1;
        end
    end
end

%% 1-D Levy process(symmetry,alpha-stable)
function y = LP(alpha,m,n,dt,C)
    V = pi/2 * (2*rand(m,n) - 1); 
    W = exprnd(1,m,n);          
%     y = sin(alpha * V) ./ ( cos(V).^(1/alpha) ) .* ...
%         ( cos( V.*(1-alpha) ) ./ W ).^( (1-alpha)/alpha ); 
    y = tan( pi/2 * (2*rand(m,n) - 1) );
    
    while(true)
        ytemp = dt^(1/alpha)*y;
        ind = find(abs(ytemp)>C);
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
function [y1,y2] = nonlocal(p,C,alpha)
    eps = 0.001;
    rmin = eps; rmax = C; thetamin = 0; thetamax = 2*pi;
    f1 = @(r,theta) (r.*cos(theta)).^2./(r.^(alpha+1));
    f2 = @(r,theta) (r.*sin(theta)).^2./(r.^(alpha+1));
    K = (alpha/(2^(1-alpha)*sqrt(pi))) * (gamma((1+alpha)/2)/gamma(1-alpha/2));
    y1 = K * integral2(f1,rmin,rmax,thetamin,thetamax);
    y2 = K * integral2(f2,rmin,rmax,thetamin,thetamax);
end



