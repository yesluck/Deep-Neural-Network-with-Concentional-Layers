clear;
close all;
clc;

%% A1:实现梯度下降法，并通过二次函数测试
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

%% TODO: 实现costFunctionA2.m, costFunctionB4.m
% 已实现
%% demo:Signal数据集实验。你不需要使用这个数据集进行实验。
% % 读取数据和标签,取前80%为训练集，后20%为测试集
% signal = load('../data/Signal/signal.mat');
% m_total = size(signal.X, 1);%总样本的个数
% n = size(signal.X,2);%样本维度
% m_train = floor( m_total*0.8 );%训练样本个数
% X_train = signal.X(1:m_train, :);%取前80%做训练数据
% y_train = signal.y(1:m_train, :);%训练标签
% X_test = signal.X(m_train+1:end, :);%取后20%做测试数据
% y_test = signal.y(m_train+1:end, :);%测试标签
% 
% %设定参数
% initial_theta = zeros(n + 1,1);
% max_iter = 100000;
% alpha = 0.001;
% lambda = 1;
% 
% % 训练demo模型
% [optimal_theta, J_history] = myfminuncA1 ( @(theta)lrCostFunction(theta,X_train,y_train,lambda), initial_theta, alpha, max_iter);
% figure;
% plot(J_history);
% title('A2 J\_history');
% 
% % 计算训练错误率
% y_train_hat = lrHypothesis(X_train, optimal_theta) >= 0.5;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('demo:train_error_rate=%f\n',train_error_rate);
% % 计算测试错误率
% y_test_hat = lrHypothesis(X_test, optimal_theta) >= 0.5;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('demo:test_error_rate=%f\n',test_error_rate);
% %将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/demo/');
% save('../output/demo/theta.mat','optimal_theta');
% save('../output/demo/param.mat','initial_theta','max_iter','alpha','lambda');%demo没有s2参数
% 

%% TODO:A3实验
tic
data=load('BankData.mat');
bankdata=data.bankdata;
m_total=size(bankdata,1);   %总样本的个数
n=size(bankdata,2)-1;       %样本维度

num_train=round(m_total*0.8);     %训练样本数
num_test=m_total-num_train;       %测试样本数
isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
train=bankdata(isTrain==1,1:n+1);     %训练样本
test =bankdata(isTrain==0,1:n+1);     %测试样本

X_train=train(:,1:n);
y_train=train(:,n+1);
X_test=test(:,1:n);
y_test=test(:,n+1);

%设定参数
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

% 训练Bankruptcy模型
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionA2(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('A3 J\_history');

% 计算训练错误率
h=hypothesisA2(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);

% 计算测试错误率
ht=hypothesisA2(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);

% %将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/Bankruptcy/');
% save('../output/Bankruptcy/theta.mat','optimal_theta');
% save('../output/Bankruptcy/param.mat','initial_theta','max_iter','alpha','lambda','s2');

toc

%% TODO:B2实验
tic

%导入数据
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
m_total = size(X,1);    % 样本总个数
n = size(X,2);          % 样本维度
num_train=round(m_total*0.8);     %训练样本数
num_test=m_total-num_train;       %测试样本数

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %训练样本
X_test =X(isTrain==0,1:n);     %测试样本
y_train=y(isTrain==1);
y_test=y(isTrain==0);

%设定参数
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

% 训练Winequality-red模型
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionA2(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('B2 J\_history');

% 计算训练错误率
h=hypothesisA2(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Winequality-red:train_error_rate=%f\n',train_error_rate);

% 计算测试错误率
ht=hypothesisA2(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Winequality-red:test_error_rate=%f\n\n',test_error_rate);

% %将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/Winequality-red/');
% save('../output/Winequality-red/theta.mat','optimal_theta');
% save('../output/Winequality-red/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demo没有s2参数

toc

%% TODO:B3实验
tic

% 导入数据
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
m_total = size(X,1);    % 样本总个数
n = size(X,2);          % 样本维度
num_train=round(m_total*0.8);     %训练样本数
num_test=m_total-num_train;       %测试样本数

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %训练样本
X_test =X(isTrain==0,1:n);     %测试样本
y_train=y(isTrain==1);
y_test=y(isTrain==0);

% 设定参数
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

% 训练MNIST模型
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionA2(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('B3 J\_history');

% 计算训练错误率
h=hypothesisA2(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('MNIST:train_error_rate=%f\n',train_error_rate);

% 计算测试错误率
ht=hypothesisA2(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('MNIST:test_error_rate=%f\n',test_error_rate);

% % 将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/MNIST/');
% save('../output/MNIST/theta.mat','optimal_theta');
% save('../output/MNIST/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demo没有s2参数

toc

%% TODO:B1比较不同的s2,alpha,lambda等参数对实验的影响
%%调整s2
% %% s2=1
% %设定参数
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
% % 训练Bankruptcy模型
% [optimal_theta, J_history1] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history1,'-g');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('s2=1:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% s2=100
% %设定参数
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
% % 训练Bankruptcy模型
% [optimal_theta, J_history100] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history100,'-r');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('s2=100:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('s2=5','s2=1','s2=100');
% 
% toc

%%调整alpha
%alpha=0.01
% %设定参数
% tic
% 
% alpha = 0.01;
% 
% % 训练Bankruptcy模型
% [optimal_theta, J_historyA] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyA,'-g');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('alpha=0.01:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
%  alpha=0.0001
% %设定参数
% tic
% 
% alpha = 0.0001;
% 
% % 训练Bankruptcy模型
% [optimal_theta, J_historyB] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyB,'-r');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('alpha=0.0001:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('alpha=0.001','alpha=0.01','alpha=0.0001');
% 
% toc

%%自编动态alpha
% %设定参数
% tic
% 
% alpha = 0.001;
% 
% % 训练Bankruptcy模型
% [optimal_theta, J_historyC] = myfminuncB1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyC,'-m');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('自编动态alpha:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% BoldDriver
% %设定参数
% tic
% 
% alpha = 0.001;
% 
% % 训练Bankruptcy模型
% [optimal_theta, J_historyD] = myfminuncBoldDriver ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyD,'-y');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('Bold Driver:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('alpha=0.001','自编动态alpha','Bold Driver');
% 
% toc

%%调整lambda
% %% lambda=100
% %设定参数
% tic
% 
% lambda = 100;
% 
% % 训练Bankruptcy模型
% [optimal_theta, J_historyA] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyA,'-g');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('lambda=100:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% lambda=0
% %设定参数
% tic
% 
% lambda = 0;
% 
% % 训练Bankruptcy模型
% [optimal_theta, J_historyB] = myfminuncA1 ( @(theta)costFunctionA2_com(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_historyB,'-r');
% 
% % 计算训练错误率
% h=hypothesisA2(X_train, optimal_theta, s2);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('lambda=0:\n');
% fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);
% 
% % 计算测试错误率
% ht=hypothesisA2(X_test, optimal_theta, s2);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);
% legend('lambda=0.001','lambda=100','lambda=0');
% 
% toc

%% B4
%使用A3实验数据集
tic
data=load('BankData.mat');
bankdata=data.bankdata;
m_total=size(bankdata,1);   %总样本的个数
n=size(bankdata,2)-1;       %样本维度

num_train=round(m_total*0.8);     %训练样本数
num_test=m_total-num_train;       %测试样本数
isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
train=bankdata(isTrain==1,1:n+1);     %训练样本
test =bankdata(isTrain==0,1:n+1);     %测试样本

X_train=train(:,1:n);
y_train=train(:,n+1);
X_test=test(:,1:n);
y_test=test(:,n+1);

%设定参数
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

% 训练Bankruptcy模型
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionB4(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('A3 J\_history');

% 计算训练错误率
h=hypothesisB4(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Bankruptcy:train_error_rate=%f\n',train_error_rate);

% 计算测试错误率
ht=hypothesisB4(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Bankruptcy:test_error_rate=%f\n',test_error_rate);

% %将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/Bankruptcy/');
% save('../output/Bankruptcy/theta.mat','optimal_theta');
% save('../output/Bankruptcy/param.mat','initial_theta','max_iter','alpha','lambda','s2');

toc

%使用B2实验数据集
tic

%导入数据
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
m_total = size(X,1);    % 样本总个数
n = size(X,2);          % 样本维度
num_train=round(m_total*0.8);     %训练样本数
num_test=m_total-num_train;       %测试样本数

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %训练样本
X_test =X(isTrain==0,1:n);     %测试样本
y_train=y(isTrain==1);
y_test=y(isTrain==0);

%设定参数
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

% 训练Winequality-red模型
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionB4(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history,'-b');
title('B2 J\_history');

% 计算训练错误率
h=hypothesisB4(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('Winequality-red:train_error_rate=%f\n',train_error_rate);

% 计算测试错误率
ht=hypothesisB4(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('Winequality-red:test_error_rate=%f\n\n',test_error_rate);

% %将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/Winequality-red/');
% save('../output/Winequality-red/theta.mat','optimal_theta');
% save('../output/Winequality-red/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demo没有s2参数

toc

%使用B3实验数据集
tic

% 导入数据
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
m_total = size(X,1);    % 样本总个数
n = size(X,2);          % 样本维度
num_train=round(m_total*0.8);     %训练样本数
num_test=m_total-num_train;       %测试样本数

isTrain=zeros(m_total,1);
isTrain(randperm(m_total,num_train),:)=1;
X_train=X(isTrain==1,1:n);     %训练样本
X_test =X(isTrain==0,1:n);     %测试样本
y_train=y(isTrain==1);
y_test=y(isTrain==0);

% 设定参数
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

% 训练MNIST模型
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionB4(theta,X_train,y_train,s2,lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history);
title('B3 J\_history');

% 计算训练错误率
h=hypothesisB4(X_train, optimal_theta, s2);
y_train_hat(h>=0.5,1)=1;
y_train_hat(h<0.5,1)=0;
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('MNIST:train_error_rate=%f\n',train_error_rate);

% 计算测试错误率
ht=hypothesisB4(X_test, optimal_theta, s2);
y_test_hat(ht>=0.5,1)=1;
y_test_hat(ht<0.5,1)=0;
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('MNIST:test_error_rate=%f\n',test_error_rate);

% % 将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/MNIST/');
% save('../output/MNIST/theta.mat','optimal_theta');
% save('../output/MNIST/param.mat','initial_theta','max_iter','alpha','lambda','s2');%demo没有s2参数

toc

%% TODO:实现C1任务，在MNIST数据集上进行实验
% max_iter = 50000;
% 
% tic
% fprintf('==========以上是MNIST对照组，以下是使用五层神经网络的实验组==========\n');
% fprintf('3层隐层，各50、40、20结点，连接函数Sigmoid、Sigmoid、ReLU、Sigmoid：\n');
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
% % m_total = size(X,1);    % 样本总个数
% % n = size(X,2);          % 样本维度
% % num_train=round(m_total*0.8);     %训练样本数
% % num_test=m_total-num_train;       %测试样本数
% % 
% % isTrain=zeros(m_total,1);
% % isTrain(randperm(m_total,num_train),:)=1;
% % X_train=X(isTrain==1,1:n);     %训练样本
% % X_test =X(isTrain==0,1:n);     %测试样本
% % y_train=y(isTrain==1);
% % y_test=y(isTrain==0);
% %%设定参数
% % 设定隐藏层层数及各层节点数
% ns = [50;40;20];
% c1 = size(ns,1);    % 隐藏层层数
% % 该隐藏层使用的激活函数：0：sigmoid函数 1：relu函数
% choise = [0,0,1,0];   % 维数必须是 c1+1 因为包含了输出层的激活函数选项
% % 计算学习参数的个数 
% s = n * ns(1);      % 输入节点到第一个隐藏层节点间变量个数（不含偏移）
% for i=1:c1-1
%     s = s + ns(i) * ns(i+1);
% end
% s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % 最后一个隐藏层到输出层和所有偏移变量
% % 随机初始化的目的是使对称失效
% initial_theta = normrnd(0,0.1,s,1);
% % 设定算法迭代次数及学习率
% % max_iter = 30000;
% alpha = 0.001;
% lambda = 0.001;
% 
% %%训练MNIST模型
% [optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise, lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history,'-r');
% title('C1 J\_history');
% 
% %%计算训练错误率
% h=hypothesisC1(X_train, optimal_theta, ns, choise);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('MNIST:train_error_rate=%f\n',train_error_rate);
% 
% %%计算测试错误率
% ht=hypothesisC1(X_test, optimal_theta, ns, choise);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
% 
% toc

% %% 4TODO:实现C1任务，在MNIST数据集上进行实验
% tic
% fprintf('==========以下是使用第一种六层神经网络的实验组==========\n');
% fprintf('4层隐层，各40、20、10、5结点，连接函数Sigmoid、ReLU、Sigmoid、ReLU、Sigmoid：\n');
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
% % m_total = size(X,1);    % 样本总个数
% % n = size(X,2);          % 样本维度
% % num_train=round(m_total*0.8);     %训练样本数
% % num_test=m_total-num_train;       %测试样本数
% % 
% % isTrain=zeros(m_total,1);
% % isTrain(randperm(m_total,num_train),:)=1;
% % X_train=X(isTrain==1,1:n);     %训练样本
% % X_test =X(isTrain==0,1:n);     %测试样本
% % y_train=y(isTrain==1);
% % y_test=y(isTrain==0);
% %%设定参数
% % 设定隐藏层层数及各层节点数
% ns = [40;20;10;5];
% c1 = size(ns,1);    % 隐藏层层数
% % 该隐藏层使用的激活函数：0：sigmoid函数 1：relu函数
% choise = [0,1,0,1,0];   % 维数必须是 c1+1 因为包含了输出层的激活函数选项
% % 计算学习参数的个数 
% s = n * ns(1);      % 输入节点到第一个隐藏层节点间变量个数（不含偏移）
% for i=1:c1-1
%     s = s + ns(i) * ns(i+1);
% end
% s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % 最后一个隐藏层到输出层和所有偏移变量
% % 随机初始化的目的是使对称失效
% initial_theta = normrnd(0,0.1,s,1);
% % 设定算法迭代次数及学习率
% max_iter = 50000;
% alpha = 0.001;
% lambda = 0.001;
% 
% %%训练MNIST模型
% [optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise, lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history,'-r');
% title('C1 J\_history');
% 
% %%计算训练错误率
% h=hypothesisC1(X_train, optimal_theta, ns, choise);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('MNIST:train_error_rate=%f\n',train_error_rate);
% 
% %%计算测试错误率
% ht=hypothesisC1(X_test, optimal_theta, ns, choise);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% %% 42TODO:实现C1任务，在MNIST数据集上进行实验
% tic
% fprintf('==========以下是使用第二种六层神经网络的实验组==========\n');
% fprintf('4层隐层，各40、20、10、5结点，连接函数ReLU、Sigmoid、ReLU、Sigmoid、Sigmoid：\n');
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
% % m_total = size(X,1);    % 样本总个数
% % n = size(X,2);          % 样本维度
% % num_train=round(m_total*0.8);     %训练样本数
% % num_test=m_total-num_train;       %测试样本数
% % 
% % isTrain=zeros(m_total,1);
% % isTrain(randperm(m_total,num_train),:)=1;
% % X_train=X(isTrain==1,1:n);     %训练样本
% % X_test =X(isTrain==0,1:n);     %测试样本
% % y_train=y(isTrain==1);
% % y_test=y(isTrain==0);
% %%设定参数
% % 设定隐藏层层数及各层节点数
% ns = [40;20;10;5];
% c1 = size(ns,1);    % 隐藏层层数
% % 该隐藏层使用的激活函数：0：sigmoid函数 1：relu函数
% choise = [1,0,1,0,0];   % 维数必须是 c1+1 因为包含了输出层的激活函数选项
% % 计算学习参数的个数 
% s = n * ns(1);      % 输入节点到第一个隐藏层节点间变量个数（不含偏移）
% for i=1:c1-1
%     s = s + ns(i) * ns(i+1);
% end
% s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % 最后一个隐藏层到输出层和所有偏移变量
% % 随机初始化的目的是使对称失效
% initial_theta = normrnd(0,0.1,s,1);
% % 设定算法迭代次数及学习率
% max_iter = 50000;
% alpha = 0.001;
% lambda = 0.001;
% 
% %%训练MNIST模型
% [optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise, lambda), initial_theta, alpha, max_iter);
% hold on;
% plot(J_history,'-g');
% title('C1 J\_history');
% 
% %%计算训练错误率
% h=hypothesisC1(X_train, optimal_theta, ns, choise);
% y_train_hat(h>=0.5,1)=1;
% y_train_hat(h<0.5,1)=0;
% train_error_rate = mean(y_train_hat ~= y_train);
% fprintf('MNIST:train_error_rate=%f\n',train_error_rate);
% 
% %%计算测试错误率
% ht=hypothesisC1(X_test, optimal_theta, ns, choise);
% y_test_hat(ht>=0.5,1)=1;
% y_test_hat(ht<0.5,1)=0;
% test_error_rate = mean(y_test_hat ~= y_test);
% fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
% 
% toc
% 
% legend('Sigmoid、Sigmoid三层神经网络','Sigmoid、ReLU、Sigmoid、ReLU、Sigmoid六层神经网络','ReLU、Sigmoid、ReLU、Sigmoid、Sigmoid六层神经网络');
% % legend('三层神经网络','七层神经网络');
% % legend('三层神经网络','五层神经网络','六层神经网络','七层神经网络');

%% 程序结束提示音
load laughter
sound(y,Fs);