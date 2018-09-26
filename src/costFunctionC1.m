function [J,grad] = costFunctionC1(theta, X, y, s, ns, choise1,choise2, lambda)
        
    %% 把待优化的参数向量theta解释为三层神经网络中的W1,b1,W2,b2
    [m,n] = size(X);                    %m：样本数，n：特征维数
    [h, W, b, a] = hypothesisC1(X, theta, ns, choise1,choise2);
    c1 = size(ns,1);    % 隐藏层层数
% h：   输出函数值
% W,b:  待训练参数
% a:    每一层输出
    %% 计算当前theta下的目标函数值    
    sum1 = 0;
    for i=1:c1+1
        sum1 = sum1 + sum(sum(W{i}.*W{i}));
    end
    J = mean( - y .* log(h) - (1 - y) .* log( 1 - h)) + lambda / 2 / m * (sum1);
    %% TODO: 计算当前theta下的目标函数梯度
    
    %%%%%%%%%%%%以下为求取与初始化神经网络基本数据的步骤%%%%%%%%%%%%
    %% 1： 计算残差
    delta = cell(c1 + 2,1);
    for j=c1+2:-1:2
        % 最后一层
        if(j==c1+2)
            if(choise1(j-1)==0)  % 最后一层传递函数为 sigmoid
                % delta{c1+2} = -y*log(h)-(1-y)*log(1-h);
                delta{c1+2} = h - y;
            else
                delta{c1+2} = -y ./ h .* relu_d(h)+ (1-y) ./(1-h) .* relu_d(1-h);
            end
        % 反向传播
        else
            if(choise1(j-1)==0)  % 最后一层激活函数为 sigmoid
                if(choise2(j)==1)
                    delta{j} = delta{j+1}*(repmat(W{j}',ns(j-1),1)').*a{j}.*(1.-a{j});
                else
                    delta{j} = delta{j+1}*(W{j}').*a{j}.*(1.-a{j});
                end
            else                % 最后一层激活函数为 relu
                if(choise2(j)==1)
                    delta{j} = delta{j+1}*repmat(W{j}',ns(j-1),1)'.*relu_d(a{j});
                else
                    delta{j} = delta{j+1}*W{j}'.*relu_d(a{j});
                end
            end
        end
    end
    
    %% 2： 计算梯度
    grad_W = cell(c1+1,1);
    grad_b = cell(c1+1,1);
    for j=1:c1+1
        if(choise2(j)==1&&j==1)
            grad_W{j} = 1/m*a{j}'*delta{j+1}+lambda/m * repmat(W{j},1,n)';
            grad_W{j} = grad_W{j}(:,1);
        elseif(choise2(j)==1)
            grad_W{j} = 1/m*a{j}'*delta{j+1}+lambda/m*repmat(W{j},1,ns(j-1))';
            grad_W{j} = grad_W{j}(:,1);
        else
            grad_W{j} = 1/m*a{j}'*delta{j+1}+lambda/m*W{j};
        end
        grad_b{j} = 1/m* sum(delta{j+1}, 1);
    end

    %% 把W, b 的偏导数排列成梯度列向量grad
    grad = [grad_W{1}(:);grad_b{1}(:)];
    for j=2:c1+1
        grad = [grad(:);grad_W{j}(:);grad_b{j}(:)];
    end
    if(s~=size(grad))
        bu = zeros(s-size(grad,1),1);
        grad = [grad(:);bu];
        %error('costFunction error!');
    end
end
