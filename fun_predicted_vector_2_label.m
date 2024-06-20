function out=fun_predicted_vector_2_label(y)
%
% predicted vector of length 10 to digit label 0, 1, ..., 10
%

[~,b]=size(y);
digit=zeros(1,b);
distance=zeros(1,b);

    for j=1:b
        I=eye(10);
        aa=zeros(1,10);
        
        for i=1:10
            aa(i)=norm(y(:,j)-I(:,i));
        end
        
        [dd,idx]=min(aa);
        
        if idx==10
            idx=0;
        end
        
        digit(j)=idx;
        distance(j)=dd;
    end
    out.label=digit;
    out.distance=distance;
end