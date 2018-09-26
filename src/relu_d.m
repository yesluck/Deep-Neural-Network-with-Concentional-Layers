function [ y ] = relu_d( X )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%   X 为列向量
y=zeros(size(X));
n1 = size(X,1);
n2 = size(X,2);
for i=1:n1
    for j=1:n2
        if(X(i,j)>0)
            y(i,j)=1;
        end
    end
end
end

