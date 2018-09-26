function [ y ] = myconv( template, target)
%	�������
%%%%%%%%%%%%%%%%%%%%%%%%% ���� %%%%%%%%%%%%%%%%%%%%%%%%%
% template: ����� n*1
% target:   ���Ŀ�� m1*m2
%%%%%%%%%%%%%%%%%%%%%%%%% ��� %%%%%%%%%%%%%%%%%%%%%%%%%
% y:        ������ m1*n
    [n,~]=size(template);
    [m1,m2]=size(target);

    y = zeros(m1,n);
    centre = round(m2/2);
    for i=1:m1
        temp = zeros(m2 + n - 1,1);
        temp(centre:(centre+n-1))=template;
        for k=1:n
            template2 = temp((k):(k+m2-1));
            y(i,k)= target(i,:) * template2;
        end
    end
end

