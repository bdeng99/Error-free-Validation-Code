%%
%
%
clear all 
% 
load ('mnist.mat')
load ('Project_1_Model_Parameters.mat')

% whos -file Project_1_Model_Parameters.mat

%%
%  input the images
%
btsz=60000;
X0=training.images(:,:,1:btsz);

a_0=fun_image_2_vector(X0); % vectorize the images for input to ANN

X0_label=training.labels(1:btsz)';


%%
% Assign the trained paraemters from Project_1_Model_Parameters.mat 
%

W1=h_nodes_40_pr_100_W.one;
W2=h_nodes_40_pr_100_W.two;
b1=h_nodes_40_pr_100_b.one;
b2=h_nodes_40_pr_100_b.two;

%%
% Evaluate the 784-n-10 model for prediction
%
predicted_vector=zeros(10,btsz);
for j=1:btsz
    z1=W1*a_0(:,j)+b1;
    [a1,~]=fun_activation(z1);

    z2=W2*a1+b2;
    [a2,~]=fun_prediction(z2);
    predicted_vector(:,j)=a2;
end

prediction=fun_predicted_vector_2_label(predicted_vector);

%%
% Compute the error and positive rates. 
%
v=abs(X0_label-prediction.label);

errt=sum(1.*(v>0))/btsz

pstvrt=(1-errt)*100
