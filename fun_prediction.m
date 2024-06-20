function [out1,out2]=fun_prediction(S)
% 
% Column-wise normaliztion
%

[out1,out2]=softmax(S);
% out2=Dsoftmax(S/bb)/bb;

    function [y,dy]=softmax(S)
        bb=1e+2;
        S=S/bb;
       aa=exp(S);
       sm=sum(exp(S));
       y=aa./sm;

       [n,~]=size(S);
       dy=zeros(n,n);
            for i=1:n
                for j=1:n
                    if j~=i
                        dy(i,j)=-exp(S(i)).*exp(S(j));
                    else
                        dy(i,j)=exp(S(i)).*sm-exp(S(i)).*exp(S(j));
                    end
                end
            end
      dy=dy/sm^2/bb;
    end
% 
%     function y=softmax(S)
%         aa=exp(S);
%         sm=sum(aa,1);
%         y=aa./sm;
%     end
end