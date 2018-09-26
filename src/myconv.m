function [ y ] = myconv( template, target)
%	卷积函数
%%%%%%%%%%%%%%%%%%%%%%%%% 输入 %%%%%%%%%%%%%%%%%%%%%%%%%
% template: 卷积核 n*1
% target:   卷积目标 m1*m2
%%%%%%%%%%%%%%%%%%%%%%%%% 输出 %%%%%%%%%%%%%%%%%%%%%%%%%
% y:        卷积结果 m1*n
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

