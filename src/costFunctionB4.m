function [J,grad] = costFunctionB4(theta, X, y, s2, lambda)
        
    %% 把待优化的参数向量theta解释为三层神经网络中的W1,b1,W2,b2
    [m,n] = size(X);                    %m：样本数，n：特征维数
    W1 = reshape(theta(1:n*s2), n, s2);
    b1 = reshape(theta(n*s2+1:(n+1)*s2), 1, s2);
    W2 = reshape(theta((n+1)*s2+1:(n+1)*s2+s2), s2, 1);     %附加说明：根据作业要求，W2维数为s2*1，故对此处代码进行修改
    b2 = theta((n+1)*s2+s2+1);                              %附加说明：根据作业要求，b2所使用的是theta向量的最后一个数（而非最后两个），故对此处代码进行修改
    %% 计算当前theta下的目标函数值    
    h = hypothesisB4(X, theta, s2); %注意，这里h返回的是一个m*1的列向量，每个元素对应一个样本
                                    %附加说明：函数增加一个自变量s2
    J = mean( - y .* log(h) - (1 - y) .* log( 1 - h)) + lambda / 2 / m * (sum(sum(W1) + sum(sum(W2))));
    %% TODO: 计算当前theta下的目标函数梯度
    grad_W1 = zeros(n, s2);
    grad_b1 = zeros(1, s2);
    grad_W2 = zeros(s2, 1);
    grad_b2 = zeros(1, 1);
    %按照向量化做法，以上步骤其实不需要了。但由于是老师程序中给出的，因此不删除了
    
    %%%%%%%%%%%%以下为求取与初始化神经网络基本数据的步骤%%%%%%%%%%%%
    a1=X;                               %Layer L1，即输入层
    a2=relu(a1*W1+repmat(b1,m,1));      %Layer L2，即隐层
    h=sigmoid(a2*W2+repmat(b2,m,1));    %Layer L3，即输出层
    
    %%%%%%%%%%%%以下为求取theta下的目标函数梯度grad的步骤%%%%%%%%%%%%
    %1、根据反向传播算法，求取输出层的delta参数(注:f'(a3)=a3.*(1.-a3))
    delta3 = h-y;
    delta2 = delta3*(W2').*relu_d(a2);
    
    %2、根据反向传播算法，计算需要的△W1、△W2、△w1、△w2
    DeltaW1=a1'*delta2;                 %求取DeltaW1，维数n*s2
    DeltaW2=a2'*delta3;                 %求取DeltaW2，维数s2*1
    Deltab1=sum(delta2,1);              %求取Deltab1
    Deltab2=sum(delta3,1);              %求取Deltab2
    
    %3、计算梯度grad
    grad_W1=1/m*DeltaW1+lambda/m*W1;
    grad_W2=1/m*DeltaW2+lambda/m*W2;
    grad_b1=1/m*Deltab1;
    grad_b2=1/m*Deltab2;
    
    %% 把W1,b1,W2,b2的偏导数排列成梯度列向量grad
    grad = [grad_W1(:); grad_b1(:); grad_W2(:); grad_b2(:)];    
end