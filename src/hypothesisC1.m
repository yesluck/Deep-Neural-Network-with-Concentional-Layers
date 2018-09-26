function [h, W, b, a] = hypothesisC1(X, theta, ns, choise1, choise2)
%%%%%%%%%%%%%%%%%%%����%%%%%%%%%%%%%%%%%%%%%%%
% X:        ����
% theta:    ��ʼ��ѵ������
% ns:       ���ز�ÿһ��Ľڵ����
% choise1:  ÿһ���ļ������ѡ��0��sigmoid 1��reLU
% choise2:  ÿһ����ȫ���Ӳ㣨0�����߾���㣨1��
%%%%%%%%%%%%%%%%%%%���%%%%%%%%%%%%%%%%%%%%%%%
% h��   �������ֵ
% W,b:  ��ѵ������
% a:    ÿһ�����

    c1 = size(ns,1);
    W = cell(c1+1,1);
    b = cell(c1+1,1);
    % �Ѵ��Ż��Ĳ�������theta����Ϊ�����������е�W1,b1,W2,b2
    [m,n] = size(X);                            % m����������n������ά��
    for i=1:c1+1
        % �����
        if(i==1)
            if(choise2(i)==0) % ȫ���Ӳ�
                k = n*ns(1);
                W{1}=reshape(theta(1:k),n,ns(1));
            else  % �����
                k = ns(1);
                W{1}=reshape(theta(1:k),ns(1),1);
            end
            b{1}=reshape(theta(k+1:k+ns(1)),1,ns(1));
            k = k+ns(1);
        % �����
        elseif(i==c1+1) 
            W{c1+1}=reshape(theta(k+1:k+ns(c1)*1),ns(c1),1);
            k = k+ns(c1)*1;
            b{c1+1} = theta(k+1);
            k = k + 1;
        % ���ز�
        else
            if(choise2(i)==0) % ȫ���Ӳ�
                W{i}=reshape(theta(k+1:k+ns(i-1)*ns(i)),ns(i-1),ns(i));
                k = k+ns(i-1)*ns(i);
            else  % �����
                W{i}=reshape(theta(k+1:k+ns(i)),ns(i),1);
                k = k+ns(i);
            end
            b{i} = reshape(theta(k+1:k+ns(i)),1,ns(i));
            k = k + ns(i);
        end
    end
    % ������δʹ����ȫ��������м��������
    if(k~=size(theta))
        error('hypothesis error!');
    end
    
    % �����ǰtheta�µ�Ŀ�꺯��ֵ
    z = cell(c1+2,1);
    a = cell(c1+2,1);
    a{1} = X;
    for i=2:c1+2
        if(choise2(i-1)==0) % ȫ���Ӳ�
            z{i} = a{i-1}*W{i-1} + b{i-1};
        else                % �����
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

