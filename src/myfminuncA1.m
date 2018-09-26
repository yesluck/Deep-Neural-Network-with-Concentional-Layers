function [optimal_theta, J_history] = myfminuncA1( costFunction, initial_theta, alpha, max_iter)
    J_history = zeros(max_iter, 1);
    theta = initial_theta;
    for i = 1:max_iter                
        [J, grad] = costFunction(theta);
        J_history(i) = J;
        theta = theta-alpha*grad;  %你应该改写为正确的梯度下降法实现
        if mod(i,100000)==0
            fprintf('已进行%d次迭代\n',i);
        end
    end
    optimal_theta = theta;    
end