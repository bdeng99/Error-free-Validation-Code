function [out1,out2]=fun_activation(S)
%
%
%

% [out1,out2]=sReLU(S);
% 
%     function [y,dy]=sReLU(x)
%         a=0.01;
%         y=(x>0).*x+(x<=0).*x*a;
% 
%         n=length(x);
%         dy=eye(n);
%         for i=1:n
%             dy(i,i)=1*(x(i)>0)+a*(x(i)<=0);
%         end
%     end

[out1,out2]=ReLU(S);

    function [y,dy]=ReLU(x)
        y=max(x,0);

        n=length(x);
        dy=eye(n);
        for i=1:n
            dy(i,i)=1*(x(i)>0);
        end
    end
end