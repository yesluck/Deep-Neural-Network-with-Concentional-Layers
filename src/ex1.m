clear;
close all;
clc;

%% A1:ʵ���ݶ��½�������ͨ�����κ�������
initial_theta = [1; 2];
max_iter = 120;
X = [3, 4];
y = 0;
alpha = 0.01;
[optimal_theta, J_history] = myfminuncA1 ( @(theta)defaultCostFunction(theta,X,y), initial_theta, alpha, max_iter);
fprintf('A1: optimal_theta=[%f,%f]\n',optimal_theta);
figure;
plot(J_history);
title('A1 J\_history');

%% TODO: ʵ��costFunctionA2.m, costFunctionB4.m
% ��ʵ��
%% demo:Signal���ݼ�ʵ�顣�㲻��Ҫʹ��������ݼ�����ʵ�顣
% % ��ȡ���ݺͱ�ǩ,ȡǰ80%Ϊѵ��������20%Ϊ���Լ�
% signal = load('../data/Signal/signal.mat');
% m_total = size(signal.X, 1);%�������ĸ���
% n = size(signal.X,2);%����ά��
% m_train = floor( m_total*0.8 );%ѵ����������
% X_train = signal.X(1:m_train, :);%ȡǰ80%��ѵ������
% y_train = signal.y(1:m_train, :);%ѵ����ǩ
% X_test = signal.X(m_train+1:end, :);%ȡ��20%����������
% y_test = signal.y(m_train+1:end, :);%���Ա�ǩ
% 
% %�趨����
% initial_theta = zeros(n + 1,1);
% max_iter = 100000;
% alpha = 0.001;
% lambda = 1;
% 
% % ѵ��demoģ��
% [optimal_theta, J_history] = myfminuncA1 ( @(theta)lrCostFunction(theta,X_train,y_train,lambda), initial_theta, alpha, max_iter);
% figure;
% plot(J_history);
% title('A2 J\_history');
% 
% % ����ѵ��������
% y_train_hat = lrHypothesis(X_train, optimal_theta) >= 0.5;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('demo:train_error_rate=%f\n',train_error_rate);
% % ������Դ�����
% y_test_hat = lrHypothesis(X_test, optimal_theta) >= 0.5;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('demo:test_error_rate=%f\n',test_error_rate);
% %��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/demo/');
% save('../output/demo/theta.mat','optimal_theta');
% save('../output/demo/param.mat','initial_theta','max_iter','alpha','lambda');%demoû��s2����
% 

%% TODO:A3ʵ��
tic
data=load('BankData.mat');
bankdata=data.bankdata;
m_total=size(bankdata,1);   %�������ĸ���
n=size(bankdata,2)-1;       %����ά��

num_train=round(m_total*0.8);     %ѵ��������
num_test=m_total-num_train;       %����������
isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
train=bankdata(isTrain==1,1:n+1);     %ѵ������
test =bankdata(isTrain==0,1:n+1);     %��������

X_train=train(:,1:n);
y_train=train(:,n+1);
X_test=test(:,1:n);
y_test=test(:,n+1);

%�趨����
s2=5;
r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(n, s2) * 2 * r - r;
W2 = rand(s2, 1) * 2 * r - r;
b1 = zeros(1, s2);
b2 = 0;
initial_theta = [W1(:); b1(:); W2(:); b2(:)];
max_iter = 100000;
alpha = 0.001;
lambda = 0.001;

% ѵ��Bankruptcyģ��
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionA2(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('A3 J\_history');

% ����ѵ��������
h=hypothesisA2(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);

% ������Դ�����
ht=hypothesisA2(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);

% %��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/Bankruptcy/');
% save('../output/Bankruptcy/theta.mat','optimal_theta');
% save('../output/Bankruptcy/param.mat','initial_theta','max_iter','alpha','lambda','s2');

toc

%% TODO:B2ʵ��
tic

%��������
red=importdata('winequality-red.csv');
X = red.data(:,1:11);
y = red.data(:,12);
for i=1:size(y,1)
    if(y(i,1)>5)
        y(i,1)=1;
    else
        y(i,1)=0;
    end
end
m_total = size(X,1);    % �����ܸ���
n = size(X,2);          % ����ά��
num_train=round(m_total*0.8);     %ѵ��������
num_test=m_total-num_train;       %����������

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %ѵ������
X_test =X(isTrain==0,1:n);     %��������
y_train=y(isTrain==1);
y_test=y(isTrain==0);

%�趨����
s2=10;
r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(n, s2) * 2 * r - r;
W2 = rand(s2, 1) * 2 * r - r;
b1 = zeros(1, s2);
b2 = 0;
initial_theta = [W1(:); b1(:); W2(:); b2(:)];
max_iter = 5000000;
alpha = 0.001;
lambda = 0;

% ѵ��Winequality-redģ��
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionA2(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('B2 J\_history');

% ����ѵ��������
h=hypothesisA2(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Winequality-red:train_error_rate=%f\n',train_error_rate);

% ������Դ�����
ht=hypothesisA2(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Winequality-red:test_error_rate=%f\n\n',test_error_rate);

% %��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/Winequality-red/');
% save('../output/Winequality-red/theta.mat','optimal_theta');
% save('../output/Winequality-red/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demoû��s2����

toc

%% TODO:B3ʵ��
tic

% ��������
data=load('mnist.mat');
X=data.X;
y=data.y;
for i=1:size(y,1)
    if(y(i,1)==4 || y(i,1)==6 || y(i,1)==8 || y(i,1)==9 || y(i,1)==10)
        y(i,1)=1;
    else
        y(i,1)=0;
    end
end
m_total = size(X,1);    % �����ܸ���
n = size(X,2);          % ����ά��
num_train=round(m_total*0.8);     %ѵ��������
num_test=m_total-num_train;       %����������

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %ѵ������
X_test =X(isTrain==0,1:n);     %��������
y_train=y(isTrain==1);
y_test=y(isTrain==0);

% �趨����
s2=150;
r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(n, s2) * 2 * r - r;
W2 = rand(s2, 1) * 2 * r - r;
b1 = zeros(1, s2);
b2 = 0;
initial_theta = [W1(:); b1(:); W2(:); b2(:)];
max_iter = 50000;
alpha = 0.001;
lambda = 0.001;

% ѵ��MNISTģ��
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionA2(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('B3 J\_history');

% ����ѵ��������
h=hypothesisA2(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('MNIST:train_error_rate=%f\n',train_error_rate);

% ������Դ�����
ht=hypothesisA2(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('MNIST:test_error_rate=%f\n',test_error_rate);

% % ��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/MNIST/');
% save('../output/MNIST/theta.mat','optimal_theta');
% save('../output/MNIST/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demoû��s2����

toc

%% TODO:B1�Ƚϲ�ͬ��s2,alpha,lambda�Ȳ�����ʵ���Ӱ��
%%����s2
% %% s2=1
% %�趨����
% tic
% 
% s2=1;
% r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
% W1 = rand(n, s2) * 2 * r - r;
% W2 = rand(s2, 1) * 2 * r - r;
% b1 = zeros(1, s2);
% b2 = 0;
% initial_theta = [W1(:); b1(:); W2(:); b2(:)];
% max_iter = 100000;
% alpha = 0.001;
% lambda = 0.001;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_history1] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history1,'-g');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('s2=1:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% s2=100
% %�趨����
% tic
% 
% s2=100;
% r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
% W1 = rand(n, s2) * 2 * r - r;
% W2 = rand(s2, 1) * 2 * r - r;
% b1 = zeros(1, s2);
% b2 = 0;
% initial_theta = [W1(:); b1(:); W2(:); b2(:)];
% max_iter = 100000;
% alpha = 0.001;
% lambda = 0.001;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_history100] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history100,'-r');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('s2=100:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('s2=5','s2=1','s2=100');
% 
% toc

%%����alpha
%alpha=0.01
% %�趨����
% tic
% 
% alpha = 0.01;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_historyA] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyA,'-g');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('alpha=0.01:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
%  alpha=0.0001
% %�趨����
% tic
% 
% alpha = 0.0001;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_historyB] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyB,'-r');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('alpha=0.0001:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('alpha=0.001','alpha=0.01','alpha=0.0001');
% 
% toc

%%�Աද̬alpha
% %�趨����
% tic
% 
% alpha = 0.001;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_historyC] = myfminuncB1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyC,'-m');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('�Աද̬alpha:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% BoldDriver
% %�趨����
% tic
% 
% alpha = 0.001;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_historyD] = myfminuncBoldDriver ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyD,'-y');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('Bold Driver:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('alpha=0.001','�Աද̬alpha','Bold Driver');
% 
% toc

%%����lambda
% %% lambda=100
% %�趨����
% tic
% 
% lambda = 100;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_historyA] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyA,'-g');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('lambda=100:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% lambda=0
% %�趨����
% tic
% 
% lambda = 0;
% 
% % ѵ��Bankruptcyģ��
% [optimal_theta, J_historyB] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyB,'-r');
% 
% % ����ѵ��������
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('lambda=0:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % ������Դ�����
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('lambda=0.001','lambda=100','lambda=0');
% 
% toc

%% B4
%ʹ��A3ʵ�����ݼ�
tic
data=load('BankData.mat');
bankdata=data.bankdata;
m_total=size(bankdata,1);   %�������ĸ���
n=size(bankdata,2)-1;       %����ά��

num_train=round(m_total*0.8);     %ѵ��������
num_test=m_total-num_train;       %����������
isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
train=bankdata(isTrain==1,1:n+1);     %ѵ������
test =bankdata(isTrain==0,1:n+1);     %��������

X_train=train(:,1:n);
y_train=train(:,n+1);
X_test=test(:,1:n);
y_test=test(:,n+1);

%�趨����
s2=5;
r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(n, s2) * 2 * r - r;
W2 = rand(s2, 1) * 2 * r - r;
b1 = zeros(1, s2);
b2 = 0;
initial_theta = [W1(:); b1(:); W2(:); b2(:)];
max_iter = 100000;
alpha = 0.001;
lambda = 0.001;

% ѵ��Bankruptcyģ��
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionB4(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('A3 J\_history');

% ����ѵ��������
h=hypothesisB4(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);

% ������Դ�����
ht=hypothesisB4(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);

% %��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/Bankruptcy/');
% save('../output/Bankruptcy/theta.mat','optimal_theta');
% save('../output/Bankruptcy/param.mat','initial_theta','max_iter','alpha','lambda','s2');

toc

%ʹ��B2ʵ�����ݼ�
tic

%��������
red=importdata('winequality-red.csv');
X = red.data(:,1:11);
y = red.data(:,12);
for i=1:size(y,1)
    if(y(i,1)>5)
        y(i,1)=1;
    else
        y(i,1)=0;
    end
end
m_total = size(X,1);    % �����ܸ���
n = size(X,2);          % ����ά��
num_train=round(m_total*0.8);     %ѵ��������
num_test=m_total-num_train;       %����������

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %ѵ������
X_test =X(isTrain==0,1:n);     %��������
y_train=y(isTrain==1);
y_test=y(isTrain==0);

%�趨����
s2=10;
r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(n, s2) * 2 * r - r;
W2 = rand(s2, 1) * 2 * r - r;
b1 = zeros(1, s2);
b2 = 0;
initial_theta = [W1(:); b1(:); W2(:); b2(:)];
max_iter = 1000000;
alpha = 0.001;
lambda = 0;

% ѵ��Winequality-redģ��
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionB4(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('B2 J\_history');

% ����ѵ��������
h=hypothesisB4(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Winequality-red:train_error_rate=%f\n',train_error_rate);

% ������Դ�����
ht=hypothesisB4(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Winequality-red:test_error_rate=%f\n\n',test_error_rate);

% %��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/Winequality-red/');
% save('../output/Winequality-red/theta.mat','optimal_theta');
% save('../output/Winequality-red/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demoû��s2����

toc

%ʹ��B3ʵ�����ݼ�
tic

% ��������
data=load('mnist.mat');
X=data.X;
y=data.y;
for i=1:size(y,1)
    if(y(i,1)==4 || y(i,1)==6 || y(i,1)==8 || y(i,1)==9 || y(i,1)==10)
        y(i,1)=1;
    else
        y(i,1)=0;
    end
end
m_total = size(X,1);    % �����ܸ���
n = size(X,2);          % ����ά��
num_train=round(m_total*0.8);     %ѵ��������
num_test=m_total-num_train;       %����������

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %ѵ������
X_test =X(isTrain==0,1:n);     %��������
y_train=y(isTrain==1);
y_test=y(isTrain==0);

% �趨����
s2=150;
r  = sqrt(6) / sqrt(n+s2+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(n, s2) * 2 * r - r;
W2 = rand(s2, 1) * 2 * r - r;
b1 = zeros(1, s2);
b2 = 0;
initial_theta = [W1(:); b1(:); W2(:); b2(:)];
max_iter = 100000;
alpha = 0.001;
lambda = 0.001;

% ѵ��MNISTģ��
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionB4(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history);
title('B3 J\_history');

% ����ѵ��������
h=hypothesisB4(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('MNIST:train_error_rate=%f\n',train_error_rate);

% ������Դ�����
ht=hypothesisB4(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('MNIST:test_error_rate=%f\n',test_error_rate);

% % ��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/MNIST/');
% save('../output/MNIST/theta.mat','optimal_theta');
% save('../output/MNIST/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demoû��s2����

toc

%% TODO:ʵ��C1������MNIST���ݼ��Ͻ���ʵ��
% max_iter = 50000;
% 
% tic
% fprintf('==========������MNIST�����飬������ʹ������������ʵ����==========\n');
% fprintf('3�����㣬��50��40��20��㣬���Ӻ���Sigmoid��Sigmoid��ReLU��Sigmoid��\n');
% 
% % data=load('mnist.mat');
% % X=data.X;
% % y=data.y;
% % for i=1:size(y,1)
% %     if(y(i,1)==4 || y(i,1)==6 || y(i,1)==8 || y(i,1)==9 || y(i,1)==10)
% %         y(i,1)=1;
% %     else
% %         y(i,1)=0;
% %     end
% % end
% % m_total = size(X,1);    % �����ܸ���
% % n = size(X,2);          % ����ά��
% % num_train=round(m_total*0.8);     %ѵ��������
% % num_test=m_total-num_train;       %����������
% % 
% % isTrain=zeros(m_total,1);
% % isTrain(randperm(m_total,num_train),:)=1;
% % X_train=X(isTrain==1,1:n);     %ѵ������
% % X_test =X(isTrain==0,1:n);     %��������
% % y_train=y(isTrain==1);
% % y_test=y(isTrain==0);
% %%�趨����
% % �趨���ز����������ڵ���
% ns = [50;40;20];
% c1 = size(ns,1);    % ���ز����
% % �����ز�ʹ�õļ������0��sigmoid���� 1��relu����
% choise = [0,0,1,0];   % ά�������� c1+1 ��Ϊ�����������ļ����ѡ��
% % ����ѧϰ�����ĸ��� 
% s = n * ns(1);      % ����ڵ㵽��һ�����ز�ڵ���������������ƫ�ƣ�
% for i=1:c1-1
%     s = s + ns(i) * ns(i+1);
% end
% s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % ���һ�����ز㵽����������ƫ�Ʊ���
% % �����ʼ����Ŀ����ʹ�Գ�ʧЧ
% initial_theta = normrnd(0,0.1,s,1);
% % �趨�㷨����������ѧϰ��
% % max_iter = 30000;
% alpha = 0.001;
% lambda = 0.001;
% 
% %%ѵ��MNISTģ��
% [optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise, lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history,'-r');
% title('C1 J\_history');
% 
% %%����ѵ��������
% h=hypothesisC1(X_train, optimal_theta, ns, choise);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('MNIST:train_error_rate=%f\n',train_error_rate);
% 
% %%������Դ�����
% ht=hypothesisC1(X_test, optimal_theta, ns, choise);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
% 
% toc

% %% 4TODO:ʵ��C1������MNIST���ݼ��Ͻ���ʵ��
% tic
% fprintf('==========������ʹ�õ�һ�������������ʵ����==========\n');
% fprintf('4�����㣬��40��20��10��5��㣬���Ӻ���Sigmoid��ReLU��Sigmoid��ReLU��Sigmoid��\n');
% 
% % data=load('mnist.mat');
% % X=data.X;
% % y=data.y;
% % for i=1:size(y,1)
% %     if(y(i,1)==4 || y(i,1)==6 || y(i,1)==8 || y(i,1)==9 || y(i,1)==10)
% %         y(i,1)=1;
% %     else
% %         y(i,1)=0;
% %     end
% % end
% % m_total = size(X,1);    % �����ܸ���
% % n = size(X,2);          % ����ά��
% % num_train=round(m_total*0.8);     %ѵ��������
% % num_test=m_total-num_train;       %����������
% % 
% % isTrain=zeros(m_total,1);
% % isTrain(randperm(m_total,num_train),:)=1;
% % X_train=X(isTrain==1,1:n);     %ѵ������
% % X_test =X(isTrain==0,1:n);     %��������
% % y_train=y(isTrain==1);
% % y_test=y(isTrain==0);
% %%�趨����
% % �趨���ز����������ڵ���
% ns = [40;20;10;5];
% c1 = size(ns,1);    % ���ز����
% % �����ز�ʹ�õļ������0��sigmoid���� 1��relu����
% choise = [0,1,0,1,0];   % ά�������� c1+1 ��Ϊ�����������ļ����ѡ��
% % ����ѧϰ�����ĸ��� 
% s = n * ns(1);      % ����ڵ㵽��һ�����ز�ڵ���������������ƫ�ƣ�
% for i=1:c1-1
%     s = s + ns(i) * ns(i+1);
% end
% s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % ���һ�����ز㵽����������ƫ�Ʊ���
% % �����ʼ����Ŀ����ʹ�Գ�ʧЧ
% initial_theta = normrnd(0,0.1,s,1);
% % �趨�㷨����������ѧϰ��
% max_iter = 50000;
% alpha = 0.001;
% lambda = 0.001;
% 
% %%ѵ��MNISTģ��
% [optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise, lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history,'-r');
% title('C1 J\_history');
% 
% %%����ѵ��������
% h=hypothesisC1(X_train, optimal_theta, ns, choise);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('MNIST:train_error_rate=%f\n',train_error_rate);
% 
% %%������Դ�����
% ht=hypothesisC1(X_test, optimal_theta, ns, choise);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% 42TODO:ʵ��C1������MNIST���ݼ��Ͻ���ʵ��
% tic
% fprintf('==========������ʹ�õڶ��������������ʵ����==========\n');
% fprintf('4�����㣬��40��20��10��5��㣬���Ӻ���ReLU��Sigmoid��ReLU��Sigmoid��Sigmoid��\n');
% 
% % data=load('mnist.mat');
% % X=data.X;
% % y=data.y;
% % for i=1:size(y,1)
% %     if(y(i,1)==4 || y(i,1)==6 || y(i,1)==8 || y(i,1)==9 || y(i,1)==10)
% %         y(i,1)=1;
% %     else
% %         y(i,1)=0;
% %     end
% % end
% % m_total = size(X,1);    % �����ܸ���
% % n = size(X,2);          % ����ά��
% % num_train=round(m_total*0.8);     %ѵ��������
% % num_test=m_total-num_train;       %����������
% % 
% % isTrain=zeros(m_total,1);
% % isTrain(randperm(m_total,num_train),:)=1;
% % X_train=X(isTrain==1,1:n);     %ѵ������
% % X_test =X(isTrain==0,1:n);     %��������
% % y_train=y(isTrain==1);
% % y_test=y(isTrain==0);
% %%�趨����
% % �趨���ز����������ڵ���
% ns = [40;20;10;5];
% c1 = size(ns,1);    % ���ز����
% % �����ز�ʹ�õļ������0��sigmoid���� 1��relu����
% choise = [1,0,1,0,0];   % ά�������� c1+1 ��Ϊ�����������ļ����ѡ��
% % ����ѧϰ�����ĸ��� 
% s = n * ns(1);      % ����ڵ㵽��һ�����ز�ڵ���������������ƫ�ƣ�
% for i=1:c1-1
%     s = s + ns(i) * ns(i+1);
% end
% s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % ���һ�����ز㵽����������ƫ�Ʊ���
% % �����ʼ����Ŀ����ʹ�Գ�ʧЧ
% initial_theta = normrnd(0,0.1,s,1);
% % �趨�㷨����������ѧϰ��
% max_iter = 50000;
% alpha = 0.001;
% lambda = 0.001;
% 
% %%ѵ��MNISTģ��
% [optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise, lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history,'-g');
% title('C1 J\_history');
% 
% %%����ѵ��������
% h=hypothesisC1(X_train, optimal_theta, ns, choise);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('MNIST:train_error_rate=%f\n',train_error_rate);
% 
% %%������Դ�����
% ht=hypothesisC1(X_test, optimal_theta, ns, choise);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% legend('Sigmoid��Sigmoid����������','Sigmoid��ReLU��Sigmoid��ReLU��Sigmoid����������','ReLU��Sigmoid��ReLU��Sigmoid��Sigmoid����������');
% % legend('����������','�߲�������');
% % legend('����������','���������','����������','�߲�������');

%% ���������ʾ��
load laughter
sound(y,Fs);