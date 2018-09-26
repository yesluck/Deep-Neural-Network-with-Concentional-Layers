%% 导入数据
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
%% 手动输入数据
% 设定隐藏层层数及各层节点数
fprintf('请输入隐藏层各层节点数：\n');
fprintf('注释说明：请以行向量的形式完成输入，行向量的长度代表隐含层个数，该行向量每一个分量对应该层层节点数！\n');
fprintf('例如：输入为[10,12],代表一共两个隐含层，第一个隐含层包含10个层节点，第二个隐含层有12个层结点，。。。\n');
ns = input('请输入隐藏层及层节点选择矩阵：');
ns = ns';
c1 = size(ns,1);    % 隐藏层层数

% 选择隐藏层使用的激活函数：0：sigmoid函数 1：relu函数
fprintf('请输入隐藏层各层激活函数：\n');
fprintf('注释说明：请以行向量的形式完成输入，行向量长度为隐含层个数加一，这是由于输出层同样需要激活函数！\n');
fprintf('该行向量对应分量只能是1和0：    0代表sigmoid函数，1代表reLU函数\n');
fprintf('例如：输入[1,0,0]，则表明在隐含层有两层的情况下，第一个隐含层使用reLU函数，第二个隐含层和输出层使用sigmoid函数\n');
choise_func = input('请输入函数选择矩阵：');
if size(choise_func,2)~=c1+1
    error('非法输入！');
end     % 维数必须是 c1+1 因为包含了输出层的激活函数选项

% 选择隐藏层的类型：0：全连接层 1：卷积层
fprintf('请输入隐藏层各层类型：\n');
fprintf('注释说明：请以行向量的形式完成输入，行向量长度为隐含层个数加一，这是由于输入层同样需要定义层类别！\n');
fprintf('该行向量对应分量只能是1和0：    0：全连接层 1：卷积层\n');
fprintf('例如：输入[0,0,0]，则表明在隐含层有两层的情况下，输入层为全连接层，第一个隐含层为全连接层、第二个隐含层为全连接层\n');
choise_lay = input('请输入层类别选择矩阵：');
%% 直接输入数据   注：此小节仅为debug使用 和上面一个小节不可同时存在
% % 设定隐藏层层数及各层节点数
% ns = [10;12];
% c1 = size(ns,1);    % 隐藏层层数
% % 该隐藏层使用的激活函数：0：sigmoid函数 1：relu函数
% choise_func = [0,1,0];   % 维数必须是 c1+1 因为包含了输出层的激活函数选项
% % 该隐藏层的类别： 0：全连接层 1：卷积层
% % 最后一层必须为全连接层 %%%% 由于包含输入层类别选择 故仍为 c1+1 维向量
% choise_lay = [0, 1, 0];
% % 计算学习参数的个数 
%% 设定参数   
% 输入节点到第一个隐藏层节点间变量个数（不含偏移）
choise_lay = zeros(size(choise_lay));
if(choise_lay(1) == 0)  % 全连接层
    s = n * ns(1); 
else                    % 卷积层
    s = ns(1);
end
for i=1:c1-1
    if(choise_lay(i+1) == 0)  % 全连接层
        s = s + ns(i) * ns(i+1);
    else                    % 卷积层
        s = s + ns(i+1);
    end
end
% 最后一层必须为全连接层 %%%%
s = s + ns(c1)*1 + sum(ns) * 1 + 1;  % 最后一个隐藏层到输出层和所有偏移变量
% 随机初始化的目的是使对称失效
initial_theta = normrnd(0,0.1,s,1);
% 设定算法迭代次数及学习率
max_iter = 5000;
alpha = 0.1;
lambda = 0.000;

%% 训练MNIST模型
[optimal_theta, J_history] = myfminuncBoldDriver ( @(theta)costFunctionC1(theta,X_train,y_train,s,ns, choise_func, choise_lay, lambda), initial_theta, alpha, max_iter);
figure;
plot(J_history);
title('C1 J\_history');

%% 计算训练错误率
h=hypothesisC1(X_train, optimal_theta, ns, choise_func, choise_lay);
y_train_hat(h>=0.5)=1;
y_train_hat(h<0.5)=0;
y_train_hat=y_train_hat';
train_error_rate = mean(y_train_hat ~= y_train);
fprintf('MNIST:train_error_rate=%f\n',train_error_rate);

%% 计算测试错误率
ht=hypothesisC1(X_test, optimal_theta, ns, choise_func, choise_lay);
y_test_hat(ht>=0.5)=1;
y_test_hat(ht<0.5)=0;
y_test_hat = y_test_hat';
test_error_rate = mean(y_test_hat ~= y_test);
fprintf('MNIST:test_error_rate=%f\n',test_error_rate);
toc;
%% 将训练结果W1,b1,W2,b2和alpha、lambda、s2等参数存盘到output下，在报告中写明存盘文件的含义。
% mkdir('../output/Bankruptcy/');
% save('../output/Bankruptcy/theta.mat','optimal_theta');
% save('../output/Bankruptcy/param.mat','initial_theta','max_iter','alpha','lambda');%demo没有s2参数