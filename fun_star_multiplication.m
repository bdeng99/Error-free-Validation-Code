function out=fun_star_multiplication(A,B)
%
%
[a,~]=size(A);
C=A;

for i=1:a
    C(i,:)=A(i,:)*B(:,:,i);
end

out=C;
% for i=1:b
%     C(:,i)=B(:,:,b)*A(:,i);
% end
% 
% out=C;
end