function h = hypothesisB4(X, theta, s2)

    % �Ѵ��Ż��Ĳ�������theta����Ϊ�����������е�W1,b1,W2,b2
    [m,n] = size(X);                    %m����������n������ά��
    W1 = reshape(theta(1:n*s2), n, s2);
    b1 = reshape(theta(n*s2+1:(n+1)*s2), 1, s2);
    W2 = reshape(theta((n+1)*s2+1:(n+1)*s2+s2), s2, 1);     %����˵����������ҵҪ��W2ά��Ϊs2*1���ʶԴ˴���������޸�
    b2 = theta((n+1)*s2+s2+1);                              %����˵����������ҵҪ��b2��ʹ�õ���theta���������һ����������������������ʶԴ˴���������޸�
    
    % �����ǰtheta�µ�Ŀ�꺯��ֵ
    a1=X;                               %Layer L1���������
    a2=relu(a1*W1+repmat(b1,m,1));      %Layer L2��������
    h=sigmoid(a2*W2+repmat(b2,m,1));    %Layer L3���������
end