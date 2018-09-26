function [J,grad] = costFunctionB4(theta, X, y, s2, lambda)
        
    %% �Ѵ��Ż��Ĳ�������theta����Ϊ�����������е�W1,b1,W2,b2
    [m,n] = size(X);                    %m����������n������ά��
    W1 = reshape(theta(1:n*s2), n, s2);
    b1 = reshape(theta(n*s2+1:(n+1)*s2), 1, s2);
    W2 = reshape(theta((n+1)*s2+1:(n+1)*s2+s2), s2, 1);     %����˵����������ҵҪ��W2ά��Ϊs2*1���ʶԴ˴���������޸�
    b2 = theta((n+1)*s2+s2+1);                              %����˵����������ҵҪ��b2��ʹ�õ���theta���������һ����������������������ʶԴ˴���������޸�
    %% ���㵱ǰtheta�µ�Ŀ�꺯��ֵ    
    h = hypothesisB4(X, theta, s2); %ע�⣬����h���ص���һ��m*1����������ÿ��Ԫ�ض�Ӧһ������
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
    a2=relu(a1*W1+repmat(b1,m,1));      %Layer L2��������
    h=sigmoid(a2*W2+repmat(b2,m,1));    %Layer L3���������
    
    %%%%%%%%%%%%����Ϊ��ȡtheta�µ�Ŀ�꺯���ݶ�grad�Ĳ���%%%%%%%%%%%%
    %1�����ݷ��򴫲��㷨����ȡ������delta����(ע:f'(a3)=a3.*(1.-a3))
    delta3 = h-y;
    delta2 = delta3*(W2').*relu_d(a2);
    
    %2�����ݷ��򴫲��㷨��������Ҫ�ġ�W1����W2����w1����w2
    DeltaW1=a1'*delta2;                 %��ȡDeltaW1��ά��n*s2
    DeltaW2=a2'*delta3;                 %��ȡDeltaW2��ά��s2*1
    Deltab1=sum(delta2,1);              %��ȡDeltab1
    Deltab2=sum(delta3,1);              %��ȡDeltab2
    
    %3�������ݶ�grad
    grad_W1=1/m*DeltaW1+lambda/m*W1;
    grad_W2=1/m*DeltaW2+lambda/m*W2;
    grad_b1=1/m*Deltab1;
    grad_b2=1/m*Deltab2;
    
    %% ��W1,b1,W2,b2��ƫ�������г��ݶ�������grad
    grad = [grad_W1(:); grad_b1(:); grad_W2(:); grad_b2(:)];    
end