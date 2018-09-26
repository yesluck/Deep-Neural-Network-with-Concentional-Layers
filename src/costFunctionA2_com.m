function [J,grad] = costFunctionA2_com(theta, X, y, s2, lambda)
        
    %% �Ѵ��Ż��Ĳ�������theta����Ϊ�����������е�W1,b1,W2,b2
    [m,n] = size(X);                    %m����������n������ά��
    W1 = reshape(theta(1:n*s2), n, s2);
    b1 = reshape(theta(n*s2+1:(n+1)*s2), 1, s2);
    W2 = reshape(theta((n+1)*s2+1:(n+1)*s2+s2), s2, 1);     %����˵����������ҵҪ��W2ά��Ϊs2*1���ʶԴ˴���������޸�
    b2 = theta((n+1)*s2+s2+1);                              %����˵����������ҵҪ��b2��ʹ�õ���theta���������һ����������������������ʶԴ˴���������޸�
    %% ���㵱ǰtheta�µ�Ŀ�꺯��ֵ    
    h = hypothesisA2(X, theta, s2); %ע�⣬����h���ص���һ��m*1����������ÿ��Ԫ�ض�Ӧһ������
                                    %����˵������������һ���Ա���s2
    J = mean( - y .* log(h) - (1 - y) .* log( 1 - h)) + lambda / 2 / m * (sum(sum(W1) + sum(sum(W2))));
    %% TODO: ���㵱ǰtheta�µ�Ŀ�꺯���ݶ�
    grad_W1 = zeros(n, s2);
    grad_b1 = zeros(1, s2);
    grad_W2 = zeros(s2, 1);
    grad_b2 = zeros(1, 1);
    %�������������������ϲ�����ʵ����Ҫ�ˡ�����������ʦ�����и����ģ���˲�ɾ����
    
    %%%%%%%%%%%%����Ϊ��ȡ���ʼ��������������ݵĲ���%%%%%%%%%%%%
    a1=X;                               %Layer L1���������
    a2=sigmoid(a1*W1+repmat(b1,m,1));   %Layer L2��������
    h=sigmoid(a2*W2+repmat(b2,m,1));    %Layer L3���������
    
    %%%%%%%%%%%%����Ϊ��ȡtheta�µ�Ŀ�꺯���ݶ�grad�Ĳ���%%%%%%%%%%%%
    %1����ȡĿ���ݶȵĹ�����p
    p=-y./h+ (1.-y)./(1.-h);
    
    %2����ȡ�ݶ�ÿ��Ԫ�صĸ�����
    qW1=a1'*((p.*h.*(1.-h))*W2'.*(a2.*(1.-a2)));
    qW2=a2'*(p.*h.*(1.-h));
    qb1=sum((p.*h.*(1.-h))*W2'.*(a2.*(1.-a2)));
    qb2=sum(p.*h.*(1.-h));

    %3�������ݶ�grad
    grad_W1=1/m*qW1+lambda/m*W1;
    grad_W2=1/m*qW2+lambda/m*W2;
    grad_b1=1/m*qb1;
    grad_b2=1/m*qb2;
    
    %% ��W1,b1,W2,b2��ƫ�������г��ݶ�������grad
    grad = [grad_W1(:); grad_b1(:); grad_W2(:); grad_b2(:)];    
end
