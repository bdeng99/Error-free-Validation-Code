function out=fun_ANN(A0,W1,b1,W2,b2)
%
%
nmb_of_lables=10;
btsz=length(A0(1,:));

    predicted_vector=zeros(nmb_of_lables,btsz);
    a_0=A0;
    for j=1:btsz
        z1=W1*a_0(:,j)+b1;
        [a1,~]=fun_activation(z1);
    
        z2=W2*a1+b2;
        [a2,~]=fun_prediction(z2);
        predicted_vector(:,j)=a2;
    end
    out=fun_predicted_vector_2_label(predicted_vector);
end