% clc;clear;
% tic
function [XX YY u] = MET2dEP(ind1,ind2,Lb1,Lb2,Ls11,Ls22,x1,x2,y1,y2,c,eps1,eps2,alpha)  
% ind1 = 1 for MET, ind1 = 0 for EP.
% ind2 = 1 for BM,ind2 = 0 for LP.
% Lb1, Lb2, Ls11 and Ls22 are the coefficients corresponding to drift and diffusion term respectively. 
% [x1,x2]*[y1,y2] is the domain in R^2.


%coefficients
drift1=@(x,y) Lb1(1) + Lb1(2)*x + Lb1(3)*y + Lb1(4)*x.^2 + Lb1(5)*x.*y + Lb1(6)*y.^2 ...
    + Lb1(7)*x.^3 + Lb1(8)*x.^2.*y + Lb1(9)*x.*y.^2 + Lb1(10)*y.^3;

drift2=@(x,y) Lb2(1) + Lb2(2)*x + Lb2(3)*y + Lb2(4)*x.^2 + Lb2(5)*x.*y + Lb2(6)*y.^2 ...
    + Lb2(7)*x.^3 + Lb2(8)*x.^2.*y + Lb2(9)*x.*y.^2 + Lb2(10)*y.^3;

diffusion1=@(x,y) Ls11(1) + Ls11(2)*x + Ls11(3)*y + Ls11(4)*x.^2 + Ls11(5)*x.*y + Ls11(6)*y.^2 ...
    + Ls11(7)*x.^3 + Ls11(8)*x.^2.*y + Ls11(9)*x.*y.^2 + Ls11(10)*y.^3;

diffusion2=@(x,y) Ls22(1) + Ls22(2)*x + Ls22(3)*y + Ls22(4)*x.^2 + Ls22(5)*x.*y + Ls22(6)*y.^2 ...
    + Ls22(7)*x.^3 + Ls22(8)*x.^2.*y + Ls22(9)*x.*y.^2 + Ls22(10)*y.^3;

% drift1=@(x,y) 3*x.^1 + 0*y.^3;
% drift2=@(x,y) 2*x + 1*y.^1;
% diffusion1=@(x,y) 1*x.^1 + 0*y;
% diffusion2=@(x,y) 0*x + 1*y.^1;
% drift1=@(x,y) 4*x - 1*x.^3;
% drift2=@(x,y) 4*y - 1*y.^3;
% diffusion1=@(x,y) 1*x.^1 + 0*y;
% diffusion2=@(x,y) 0*x + 1*y.^1;

%boundary
% x1 = 1;x2 = 2;
% y1 = 1;y2 = 2;
y3 = 3;
dx = 0.01;dy = 0.01;

% 先i,后j
x = x1:dx:x2;
y = y1:dy:y2;
m = length(x);
n = length(y);

%calculating coefficients
f = zeros((m-2)*(n-2),1);
g = f;sigma1 = f;sigma2 = f;
u =zeros(m,n);
count = 1;
for j = 2:n-1
    for i = 2:m-1
        f(count) = drift1(x(i),y(j));
        g(count) = drift2(x(i),y(j));
        sigma1(count) = diffusion1(x(i),y(j)).^2;
        sigma2(count) = diffusion2(x(i),y(j)).^2;
        count = count + 1;
    end
end
s1 = 0.5*sigma1/(dx^2) + 0.5*f/dx;%C1
s2 = sigma1/(dx^2) + sigma2/(dy^2); %B1+B2
s3 = 0.5*sigma1/(dx^2) - 0.5*f/dx; %A1
s4 = 0.5*sigma2/(dy^2) + 0.5*g/dy; %C2
s5 = 0.5*sigma2/(dy^2) - 0.5*g/dy;%A2
s1(n-2:n-2:(n-2)^2) = 0;
s3(n-1:n-2:(n-2)^2) = 0;

%drift part and BM part
A=diag(-s2)+diag(s1(1:end-1),1)+diag(s3(2:end),-1)+diag(s4(1:end-(n-2)),n-2)+diag(s5(n-2+1:end),-(n-2));  %% differential matrix
% spy(A(80:120,80:120))

%pure jump Levy part
DD = zeros((m-2)*(n-2));
if  ind2 == 0
    D1 = zeros(m*n);
    D2 = D1;D3 = D1;D4 = D1;  %% integral matrix
    h = dx;
    % c = 100; %jump bound
    % eps1 =1.; %Levy noise intensity
    % eps2 = 1.; %Levy noise intensity
    % alpha = 1; %stable index
    K = (alpha/(2^(1-alpha)*sqrt(pi))) * (gamma((1+alpha)/2)/gamma(1-alpha/2)); %% Jump measure constant
    C1 = (eps1^alpha)*K*h; C2 = (eps2^alpha)*K*h;
    %x-direction
    for i = 1:(m)*(n)
        if i == 1 || i == m*n || mod(i,n) == 1 || mod(i,n) == 0  %boundary condition = 0
            continue
        end
        k = mod(i,m);%求余 i
        if k == 0
            z = ((1-n)* h):h:((x2-x1)+(1-n)*h);
        else
            z = ((1-k)* h):h:((x2-x1)+(1-k)*h);
        end
        ind = floor((i-0.01)/m);%向下取整 j
        for j = 2:m-1
            if abs(z(j)) >= 0.01 && abs(z(j)) <= c 
                D1(i,ind*n+j) = C1/(abs(z(j))^(1+alpha));
                if k == 0
                    D2(i,ind*n+m) = D2(i,ind*n+m) + C1/(abs(z(j))^(1+alpha));
                    disp('I am here')
                else
                    D2(i,ind*n+k) = D2(i,ind*n+k) + C1/(abs(z(j))^(1+alpha));
                end
            else
                D1(i,ind*n+j) = 0;
            end
        end
    end
    d = zeros(m*n,1);%new
    %y-direction
    for i = 1:(m)*(n)
        if i == 1 || i == m*n || mod(i,n) == 1 || mod(i,n) == 0  %boundary condition = 0
            continue
        end
        k = mod(i,m);%求余 i
        if k == 0
            z = ((1-n)* h):h:((y2-y1)+(1-n)*h);
            ztemp = ((y2-y1)+(1+1-n)*h):h:((y3-y1)+(1-n)*h); %new
        else
            z = ((1-k)* h):h:((y2-y1)+(1-k)*h);
            ztemp = ((y2-y1)+(1+1-k)*h):h:((y3-y1)+(1-k)*h); %new
        end
        ind = floor((i-0.01)/m);%向下取整 j
        for j = 1:n-2
            %new--------%
            if abs(ztemp(j))>=0.01 && abs(ztemp(j))<=c
                d(i) = d(i) + C2/(abs(ztemp(j))^(1+alpha));
            else
                d(i) = 0;
            end
            if j == n-2
                d(i) = d(i) + C2/(abs(ztemp(j+1))^(1+alpha));
            end
            %--------%

            if abs(z(j)) >= 0.01 && abs(z(j)) <= c 
                D3(i,j*m+k) = C2/(abs(z(j))^(1+alpha));
                if k == 0
                    D4(i,j*m+m) = D4(i,j*m+m) + C2/(abs(z(j))^(1+alpha));
                    disp('I am here y')
                else
                    D4(i,j*m+k) = D4(i,j*m+k) + C2/(abs(z(j))^(1+alpha));
                end
            else
    %             D3(i,ind*n+j) = 0;
                D3(i,j*m+k) = 0;
    %             disp('I am here yy')
            end
        end
    end
    Dtemp = (D1-D2) + (D3-D4);

    %除去0边界  求EP时要小心这里
    for i = 2:m-1
        for j = 2:n-1
            DD((i-2)*(m-2)+1:(i-2)*(m-2)+m-2,(j-2)*(n-2)+1:(j-2)*(n-2)+n-2) = ...
                Dtemp((i-1)*(m)+2:(i-1)*(m)+m-1,(j-1)*(n)+2:(j-1)*(n)+n-1);
        end
    end
end
D = A + 1*DD;  %线性方程组矩阵

%MET
if ind1 == 1 %MET
    yy = -ones((m-2)*(n-2),1);
    utemp = D \ yy; %求解线性方程组
end

%EP
if ind1 == 0 %EP
    count = 1;
    if ind2 == 0 %Levy Process
        for i = 1:m*n
            if (i>=1 && i <= m) || (i>=m*(n-1)+1 && i <= m*n)
                continue
            end
            k = mod(i,m);
            if k == 1 || k == 0
                continue
            else
                dd(count) =  d(i);
                count = count + 1;
            end
        end
    end
    yy = zeros((m-2)*(n-2),1);
%     yy(1:m-2) = -s5(1:m-2);
    yy(end-(m-3):end) = -s4(end-(m-3):end);
    if ind2 == 0 %Levy Process
%         yy(1:m-2) = yy(1:m-2) - D3(2:m-1,n);
        yy(end-(m-3):end) = yy(end-(m-3):end) - D3(end-(m-2):end-1,n);
        yy = yy - dd';
    end
    utemp = D \ yy; %求解线性方程组
    u(2:m-1,n) = 1;
end


for i = 2:n-1
    u(2:m-1,i) = utemp(1+(i-2)*(m-2):(i-1)*(m-2));
end

%plot
[XX,YY] = meshgrid(x,y);
% figure
% mesh(XX,YY,u')
% xlabel('x')
% ylabel('y')
% zlabel('u')
% axis([x1 x2 0 0.3])
% title('Mean Exit Time')
% legend('true','learn')
% toc