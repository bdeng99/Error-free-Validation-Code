%%
% plot clustering demo
%
selected_image_labels=training.labels(data_seq)';

nmb_of_cluster_label=60;
cluster_1=fun_label_class(1,selected_image_labels,nmb_of_cluster_label);
cluster_2=fun_label_class(2,selected_image_labels,nmb_of_cluster_label);
cluster_3=fun_label_class(3,selected_image_labels,nmb_of_cluster_label);
%%
T01=T0(:,cluster_1);
T02=T0(:,cluster_2);
T03=T0(:,cluster_3);
A=[[0;1] [-sqrt(3)/2;-1/2] [sqrt(3)/2;-1/2]];
ms=8;
uu=1:3;

%%
% clustering plot
%
% Compare SGD and GDT results
%
load SGD_Trained_Model.mat W1 b1 W2 b2

figure(4)
hold off
for x=1
fs=12;
vv=randperm(length(T0(:,1)));
v=vv(uu);
v =[384   438   212];

a01=A*T01(v,:);
a02=A*T02(v,:);
a03=A*T03(v,:);

subplot(2,3,1)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Initial Clustering: a^0','fontsize',fs)

%%%%%%%
% SGD: a^1 transd image
%

a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

a01=W2*a01+b2;
a02=W2*a02+b2;
a03=W2*a03+b2;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,2)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('SGD Affine Map 2: z_2=W_2*a_1`+b_2','fontsize',fs)

%%%%%%%
% SGD: a^2 transd image
%
a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

scl=1;
a01=W2*a01+scl*b2;
a02=W2*a02+scl*b2;
a03=W2*a03+scl*b2;

for jj=1:nmb_of_cluster_label
a01(:,jj)=fun_prediction(a01(:,jj));
a02(:,jj)=fun_prediction(a02(:,jj));
a03(:,jj)=fun_prediction(a03(:,jj));
end

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,3)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('SGD Activation 2: a_2 = softmax(z_2)','fontsize',fs)

%%%%%%
load GDT_Trained_Model.mat W1 b1 W2 b2

%%%%%%%%%%
% GDT: z^1 transd image
%
a01=W1*T01+b1;
a02=W1*T02+b1;
a03=W1*T03+b1;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,4)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('GDT Affine Map 1: z_1 = W_1*a^0+b_1','fontsize',fs)

%
% GDT: z^2 transd image
%
a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

a01=W2*a01+b2;
a02=W2*a02+b2;
a03=W2*a03+b2;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,5)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('GDT Affine Map 2: z_2 = W_2*a_1`+b_2','fontsize',fs)

%
% GDT: a^2 transd image
% 
a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

scl=1;
a01=W2*a01+scl*b2;
a02=W2*a02+scl*b2;
a03=W2*a03+scl*b2;

for jj=1:nmb_of_cluster_label
a01(:,jj)=fun_prediction(a01(:,jj));
a02(:,jj)=fun_prediction(a02(:,jj));
a03(:,jj)=fun_prediction(a03(:,jj));
end

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('GDT Activation 2: a_2 = softmax(z_2)','fontsize',fs)

set(gcf,'Position',[10 10 1200 700])
set(gcf, 'MenuBar', 'None')
end

return
%%
load GDT_Trained_Model.mat W1 b1 W2 b2 
% load SGD_Trained_Model.mat W b

figure(5)
hold off
for x=1
fs=12;
vv=randperm(length(T0(:,1)));
v=vv(uu);
v =[384   438   212];

a01=A*T01(v,:);
a02=A*T02(v,:);
a03=A*T03(v,:);

subplot(2,3,1)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Initial Clustering: a^0','fontsize',fs)

%
% W1 transd image
%
% a01=W1*T0(:,cluster_1);
% a02=W1*T0(:,cluster_2);
% a03=W1*T0(:,cluster_3);
% 
% a01=A*a01(uu,:);
% a02=A*a02(uu,:);
% a03=A*a03(uu,:);

% subplot(2,3,2)
% hold off
% 
% clustering_plot
% % axis([-inf inf -inf inf])
% title('Linear Map: a^0 to W_1*a^0','fontsize',fs)
%
% W1 + b1 transd image
%
a01=W1*T01+b1;
a02=W1*T02+b1;
a03=W1*T03+b1;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,2)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Affine Map 1: z_1=W_1*a^0`+b_1','fontsize',fs)
%
% a^1 transd image
%
a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,3)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Activation 1: a_1=ReLU(z_1)','fontsize',fs)

%
% Linear Map 2 transd image
%
a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

scl=0;
a01=W2*a01+scl*b2;
a02=W2*a02+scl*b2;
a03=W2*a03+scl*b2;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,4)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Linear Map 2: a_1 to W_2*a_1','fontsize',fs)

%
% z^2 transd image
%
a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

a01=W2*a01+b2;
a02=W2*a02+b2;
a03=W2*a03+b2;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,5)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Affine Map 2: z_2=W_2*a_1`+b_2','fontsize',fs)

%
% a^2 transd image
%
a01=fun_activation(W1*T01+b1);
a02=fun_activation(W1*T02+b1);
a03=fun_activation(W1*T03+b1);

scl=1;
a01=W2*a01+scl*b2;
a02=W2*a02+scl*b2;
a03=W2*a03+scl*b2;

for jj=1:nmb_of_cluster_label
a01(:,jj)=fun_prediction(a01(:,jj));
a02(:,jj)=fun_prediction(a02(:,jj));
a03(:,jj)=fun_prediction(a03(:,jj));
end

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

subplot(2,3,6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Activation 2: a_2=softmax(z_2)','fontsize',fs)

set(gcf,'Position',[10 10 1200 700])
set(gcf, 'MenuBar', 'None')
end

%%
% initial image
%
vv=randperm(length(T0(:,1)));
v=vv(uu);
v =[384   438   212];

a01=A*T01(v,:);
a02=A*T02(v,:);
a03=A*T03(v,:);

figure(6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Initial Clustering: a^0','fontsize',16)

%%
% W1 transd image
%
a01=W1*T0(:,cluster_1);
a02=W1*T0(:,cluster_2);
a03=W1*T0(:,cluster_3);

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

figure(6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Linear Map: a^0 -> W_1*a^0','fontsize',16)
%%
% W1 + b1 transd image
%
a01=W1*T0(:,cluster_1)+b1;
a02=W1*T0(:,cluster_2)+b1;
a03=W1*T0(:,cluster_3)+b1;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

figure(6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Affine Map 1: z_1=W_1*a^0`+b_1','fontsize',16)
%%
% a^1 transd image
%
a01=fun_activation(W1*T0(:,cluster_1)+b1);
a02=fun_activation(W1*T0(:,cluster_2)+b1);
a03=fun_activation(W1*T0(:,cluster_3)+b1);

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

figure(6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Activation 1: a_1=ReLU(z_1)','fontsize',16)
%%
% z^2 transd image
%
a01=fun_activation(W1*T0(:,cluster_1)+b1);
a02=fun_activation(W1*T0(:,cluster_2)+b1);
a03=fun_activation(W1*T0(:,cluster_3)+b1);

a01=W2*a01+b2;
a02=W2*a02+b2;
a03=W2*a03+b2;

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

figure(6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Affine Map 2: z_2=W_2*a_1`+b_2','fontsize',16)
%%
% a^2 transd image
%
a01=fun_activation(W1*T0(:,cluster_1)+b1);
a02=fun_activation(W1*T0(:,cluster_2)+b1);
a03=fun_activation(W1*T0(:,cluster_3)+b1);


a01=W2*a01+b2;
a02=W2*a02+b2;
a03=W2*a03+b2;

for jj=1:nmb_of_cluster_label
a01(:,jj)=fun_prediction(a01(:,jj));
a02(:,jj)=fun_prediction(a02(:,jj));
a03(:,jj)=fun_prediction(a03(:,jj));
end

a01=A*a01(uu,:);
a02=A*a02(uu,:);
a03=A*a03(uu,:);

figure(6)
hold off

clustering_plot
% axis([-inf inf -inf inf])
title('Activation 2: a_2=softmax(z_2)','fontsize',16)

