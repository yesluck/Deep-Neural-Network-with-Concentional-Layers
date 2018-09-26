function h = hypothesisB4(X, theta, s2)

    % 把待优化的参数向量theta解释为三层神经网络中的W1,b1,W2,b2
    [m,n] = size(X);                    %m：样本数，n：特征维数
    W1 = reshape(theta(1:n*s2), n, s2);
    b1 = reshape(theta(n*s2+1:(n+1)*s2), 1, s2);
    W2 = reshape(theta((n+1)*s2+1:(n+1)*s2+s2), s2, 1);     %附加说明：根据作业要求，W2维数为s2*1，故对此处代码进行修改
    b2 = theta((n+1)*s2+s2+1);                              %附加说明：根据作业要求，b2所使用的是theta向量的最后一个数（而非最后两个），故对此处代码进行修改
    
    % 输出当前theta下的目标函数值
    a1=X;                               %Layer L1，即输入层
    a2=relu(a1*W1+repmat(b1,m,1));      %Layer L2，即隐层
    h=sigmoid(a2*W2+repmat(b2,m,1));    %Layer L3，即输出层
end