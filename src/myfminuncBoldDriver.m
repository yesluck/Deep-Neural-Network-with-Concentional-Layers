function [optimal_theta, J_history] = myfminuncBoldDriver( costFunction, initial_theta, alpha, max_iter)
    J_history = zeros(max_iter, 1);
    theta = initial_theta;
    for i = 1:max_iter                
        [J, grad] = costFunction(theta);
        J_history(i) = J;
        
        if i==10
            
        end
        
        if i>=2 && J<=J_history(i-1)
            alpha=1.05*alpha;
        end
        if i>=2 && J>J_history(i-1)
            alpha=0.5*alpha;
        end
        
        if J<=0.000005 || isnan(J)==true
            fprintf('�ڵ�%d�ε����У�J=%f��С����ֵ���Զ���ֹ����\n',i,J);
            toc
            tic
            break;
        end
        
        theta = theta-alpha*grad;
        
        if mod(i,round(max_iter/5))==0 && i>=10000
            fprintf('�ѽ���%d�ε���,J=%f\n',i,J);
            toc
            tic
        end
    end
    optimal_theta = theta;    
end