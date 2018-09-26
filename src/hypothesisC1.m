function [h, W, b, a] = hypothesisC1(X, theta, ns, choise1, choise2)
%%%%%%%%%%%%%%%%%%%输入%%%%%%%%%%%%%%%%%%%%%%%
% X:        样本
% theta:    初始待训练参数
% ns:       隐藏层每一层的节点个数
% choise1:  每一级的激活函数的选择：0：sigmoid 1：reLU
% choise2:  每一层是全连接层（0）或者卷积层（1）
%%%%%%%%%%%%%%%%%%%输出%%%%%%%%%%%%%%%%%%%%%%%
% h：   输出函数值
% W,b:  待训练参数
% a:    每一层输出

    c1 = size(ns,1);
    W = cell(c1+1,1);
    b = cell(c1+1,1);
    % 把待优化的参数向量theta解释为三层神经网络中的W1,b1,W2,b2
    [m,n] = size(X);                            % m：样本数，n：特征维数
    for i=1:c1+1
        % 输入层
        if(i==1)
            if(choise2(i)==0) % 全连接层
                k = n*ns(1);
                W{1}=reshape(theta(1:k),n,ns(1));
            else  % 卷积层
                k = ns(1);
                W{1}=reshape(theta(1:k),ns(1),1);
            end
            b{1}=reshape(theta(k+1:k+ns(1)),1,ns(1));
            k = k+ns(1);
        % 输出层
        elseif(i==c1+1) 
            W{c1+1}=reshape(theta(k+1:k+ns(c1)*1),ns(c1),1);
            k = k+ns(c1)*1;
            b{c1+1} = theta(k+1);
            k = k + 1;
        % 隐藏层
        else
            if(choise2(i)==0) % 全连接层
                W{i}=reshape(theta(k+1:k+ns(i-1)*ns(i)),ns(i-1),ns(i));
                k = k+ns(i-1)*ns(i);
            else  % 卷积层
                W{i}=reshape(theta(k+1:k+ns(i)),ns(i),1);
                k = k+ns(i);
            end
            b{i} = reshape(theta(k+1:k+ns(i)),1,ns(i));
            k = k + ns(i);
        end
    end
    % 若参数未使用完全，则表明中间运算出错
    if(k~=size(theta))
        error('hypothesis error!');
    end
    
    % 输出当前theta下的目标函数值
    z = cell(c1+2,1);
    a = cell(c1+2,1);
    a{1} = X;
    for i=2:c1+2
        if(choise2(i-1)==0) % 全连接层
            z{i} = a{i-1}*W{i-1} + b{i-1};
        else                % 卷积层
            z{i} = myconv(W{i-1},a{i-1}) + b{i-1};
        end
        if(choise1(i-1)==0) 
            a{i} = sigmoid(z{i});
        else
            a{i} = relu(z{i});
        end
    end
    h = a{c1+2};
end

