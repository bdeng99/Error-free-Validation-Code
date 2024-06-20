function v=fun_image_2_vector(A)
%
%
    n=length(A(1,1,:));
    [a,b]=size(A(:,:,1));
    v=zeros(a*b,n);
    for i=1:n
        a=A(:,:,i);
        v(:,i)=a(:);
    end
end