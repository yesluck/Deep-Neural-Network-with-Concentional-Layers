function [optimal_theta, J_history] = myfminuncA1( costFunction, initial_theta, alpha, max_iter)
    J_history = zeros(max_iter, 1);
    theta = initial_theta;
    for i = 1:max_iter                
        [J, grad] = costFunction(theta);
        J_history(i) = J;
        theta = theta-alpha*grad;  %��Ӧ�ø�дΪ��ȷ���ݶ��½���ʵ��
        if mod(i,100000)==0
            fprintf('�ѽ���%d�ε���\n',i);
        end
    end
    optimal_theta = theta;    
end