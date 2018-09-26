%% ��������
clear,clc;
tic;
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
%% �ֶ���������
% �趨���ز����������ڵ���
fprintf('���������ز����ڵ�����\n');
fprintf('ע��˵������������������ʽ������룬�������ĳ��ȴ����������������������ÿһ��������Ӧ�ò��ڵ�����\n');
fprintf('���磺����Ϊ[10,12],����һ�����������㣬��һ�����������10����ڵ㣬�ڶ�����������12�����㣬������\n');
ns = input('���������ز㼰��ڵ�ѡ�����');
ns = ns';
c1 = size(ns,1);    % ���ز����

% ѡ�����ز�ʹ�õļ������0��sigmoid���� 1��relu����
fprintf('���������ز���㼤�����\n');
fprintf('ע��˵������������������ʽ������룬����������Ϊ�����������һ���������������ͬ����Ҫ�������\n');
fprintf('����������Ӧ����ֻ����1��0��    0����sigmoid������1����reLU����\n');
fprintf('���磺����[1,0,0]��������������������������£���һ��������ʹ��reLU�������ڶ���������������ʹ��sigmoid����\n');
choise_func = input('�����뺯��ѡ�����');
if size(choise_func,2)~=c1+1
    error('�Ƿ����룡');
end     % ά�������� c1+1 ��Ϊ�����������ļ����ѡ��

% ѡ�����ز�����ͣ�0��ȫ���Ӳ� 1�������
fprintf('���������ز�������ͣ�\n');
fprintf('ע��˵������������������ʽ������룬����������Ϊ�����������һ���������������ͬ����Ҫ��������\n');
fprintf('����������Ӧ����ֻ����1��0��    0��ȫ���Ӳ� 1�������\n');
fprintf('���磺����[0,0,0]��������������������������£������Ϊȫ���Ӳ㣬��һ��������Ϊȫ���Ӳ㡢�ڶ���������Ϊȫ���Ӳ�\n');
choise_lay = input('����������ѡ�����');
%% ֱ����������   ע����С�ڽ�Ϊdebugʹ�� ������һ��С�ڲ���ͬʱ����
% % �趨���ز����������ڵ���
% ns = [10;12];
% c1 = size(ns,1);    % ���ز����
% % �����ز�ʹ�õļ������0��sigmoid���� 1��relu����
% choise_func = [0,1,0];   % ά�������� c1+1 ��Ϊ�����������ļ����ѡ��
% % �����ز����� 0��ȫ���Ӳ� 1�������
% % ���һ�����Ϊȫ���Ӳ� %%%% ���ڰ�����������ѡ�� ����Ϊ c1+1 ά����
% choise_lay = [0, 1, 0];
% % ����ѧϰ�����ĸ��� 
%% �趨����   
% ����ڵ㵽��һ�����ز�ڵ���������������ƫ�ƣ�
choise_lay = zeros(size(choise_lay));
if(choise_lay(1) == 0)  % ȫ���Ӳ�
    s = n * ns(1); 
else                    % �����
    s = ns(1);
end
for i=1:c1-1
    if(choise_lay(i+1) == 0)  % ȫ���Ӳ�
        s = s + ns(i) * ns(i+1);
    else                    % �����
        s = s + ns(i+1);
    end
end
% ���һ�����Ϊȫ���Ӳ� %%%%
s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % ���һ�����ز㵽����������ƫ�Ʊ���
% �����ʼ����Ŀ����ʹ�Գ�ʧЧ
initial_theta = normrnd(0,0.1,s,1);
% �趨�㷨����������ѧϰ��
max_iter = 5000;
alpha = 0.1;
lambda = 0.000;

%% ѵ��MNISTģ��
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise_func, choise_lay, lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history);
title('C1 J\_history');

%% ����ѵ��������
h=hypothesisC1(X_train, optimal_theta, ns, choise_func, choise_lay);
y_train_hat(h>=0.5)=1;
y_train_hat(h<0.5)=0;
y_train_hat=y_train_hat';
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('MNIST:train_error_rate=%f\n',train_error_rate);

%% ������Դ�����
ht=hypothesisC1(X_test, optimal_theta, ns, choise_func, choise_lay);
y_test_hat(ht>=0.5)=1;
y_test_hat(ht<0.5)=0;
y_test_hat = y_test_hat';
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
toc;
%% ��ѵ�����W1,b1,W2,b2��alpha��lambda��s2�Ȳ������̵�output�£��ڱ�����д�������ļ��ĺ��塣
% mkdir('../output/Bankruptcy/');
% save('../output/Bankruptcy/theta.mat','optimal_theta');
% save('../output/Bankruptcy/param.mat','initial_theta','max_iter','alpha','lambda');%demoû��s2����