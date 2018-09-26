function [optimal_theta, J_history] = myfminuncB1( costFunction, initial_theta, alpha, max_iter)
    J_history = zeros(max_iter, 1);
    theta = initial_theta;
    for i = 1:max_iter                
        [J, grad] = costFunction(theta);
        J_history(i) = J;
        if i<round(max_iter/5)
            theta = theta-10*alpha*grad;  %你应该改写为正确的梯度下降法实现
        else
            if i>round(4*max_iter/5)
                theta = theta-0.1*alpha*grad;
            else
                theta = theta-alpha*grad;
            end
        end
        if mod(i,100000)==0
            fprintf('已进行%d次迭代\n',i);
        end
    end
    optimal_theta = theta;    
end