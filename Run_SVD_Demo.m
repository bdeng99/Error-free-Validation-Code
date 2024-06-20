
clear all 

load ('mnist.mat')

% % Find out what is in the file
%  whos -filemnist.mat
% % enter the variable name to find out its structure
% test
% % it says test has two more variable: test.images and test.labels
% % the same for the other variable 'training'
% training
% % training.labels(9)


% Let's say we have 2000 training samples that we are going to use .

% We can divide the dataset of 2000 into batches of 500 
% then it will take 4 iterations to complete 1 epoch.

%%
%  input the images
%
data_size=1000; % 60000; % for full training images
aa=1:60000;%randperm(60000);
aa=randperm(60000);
data_seq=aa(1:data_size);
T0=fun_image_2_vector(training.images(:,:,data_seq));

T0_label=training.labels(data_seq)';
image_labels=T0_label;
idx=(image_labels==0);
image_labels(idx)=10;


nmb_of_labels=10;

training_vectors=fun_classification_label2vector(image_labels,nmb_of_labels); 

%%
Classification_matrix=training_vectors*pinv(T0);


%%
sum(sum(abs(Classification_matrix*T0-training_vectors)))

%%

A=[[1 0 0 1 0];[2 2 2 2 1];[3 1 3 3 1];[3 0 0 3 1]];

[U, S, V]=svd(A);
% 
% 
% %%
% pinv(A)*A

Sinv=pinv(S);
Aright=V*Sinv*U';
% Aleft=V*Sinv*U';
% A*V*pinv(S)*U';

B=[eye(4),[0;1;0;0]];

C=B*Aright;
norm(C*A-B)


norm(A-U*S*V')
%%
D=U'*U;
D=Sinv*D;
D=D*S;
D=D*V';
D=V*D