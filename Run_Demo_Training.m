% 
%  Error-free training for ANN
%
%  Bo Deng
%  Dept. of Math. 
%  UNL, Lincoln, NE 68605
%  
%  For education only. All rights reserved.
%
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
% return
%%
% basic model paraemters for 784-n-10 ANN
%
input_size=length(T0(:,1)); % 784
nmb_of_lables=10;
nmb_of_h_nodes=80; % nmb of hidden nodes

ANN_parameters.display_number=25;
ANN_parameters.image_size=[28,28];
ANN_parameters.A0_labels_in_number=T0_label;
ANN_parameters.selected_images_relative_location=1:data_size;

aa=T0_label;
idx=(aa==0);
aa(idx)=10;
ANN_parameters.A0_labels_in_number_position=aa;

%%
%
%  Initialize all training parameters p={W,b}.
%

int_scl_para=10;
cntr=0.5;

W1=int_scl_para*(rand(nmb_of_h_nodes,input_size)-cntr);
W2=int_scl_para*(rand(nmb_of_lables,nmb_of_h_nodes)-cntr);
b1=int_scl_para*(rand(nmb_of_h_nodes,1)-cntr);
b2=int_scl_para*(rand(nmb_of_lables,1)-cntr);

save Untrained_Model.mat W1 b1 W2 b2

%%
%%%%%%%%%%%%%%%%%%%%%
% Plot the performance of the initialized model
%

    nmb_of_plot_image=min(ANN_parameters.display_number, data_size);
    selected_sample_seq=randperm(nmb_of_plot_image);

    predicted=fun_ANN(T0,W1,b1,W2,b2);

    figure_number=10;
    ANN_parameters.nmb_of_learning_sessions=0; 
    fun_plot_input_vs_output(T0,ANN_parameters, predicted.label,figure_number, selected_sample_seq)
    sgtitle("ANN Model: Input vs Output (Untrained, Training Session = " + 0 + ")")

% xshft=2000;
% yshft=50;
%%%%%%%laptop setting
xshft=1800;
yshft=-308;
%%%%%%%laptop setting

set(gcf,'Position',[-1800+xshft 300+yshft 1000 290])
set(gcf, 'MenuBar', 'None')

figure(10)
hold off

%%
% Stochastic Gradient Descent to prime the search
%

% %%%%%%%% for 1000 demo
% nmb_of_searching_per_session=2800;
% nmb_of_sessions=1;
% random_batch_size=1:800;

%%%%%%%% for <=1000 demo
nmb_of_searching_per_session=60;
nmb_of_sessions=40;
random_batch_size=1:70;

% %%%%%%% for full 60000
% nmb_of_searching_per_session=80;
% nmb_of_sessions=4000;
% random_batch_size=1:1000;

SGD_training_positive_rate=zeros(1,nmb_of_sessions);

SGD_parameters=ANN_parameters;
SGD_parameters.nmb_of_gradient_decent_iterations=nmb_of_searching_per_session;
SGD_parameters.nmb_of_learning_sessions=0;

for kk=1:nmb_of_sessions

    SGD_parameters.nmb_of_learning_sessions=[SGD_parameters.nmb_of_learning_sessions,kk];

    %%%%%%% Compute the postive rate for W,b
            predicted=fun_ANN(T0,W1,b1,W2,b2);
            SGD_training_positive_rate(kk)=100*(data_size-sum(1*(abs(T0_label-predicted.label)>0)))/data_size;

%%%%%%% New training session
%
    rp=randperm(data_size); 
    sgd_batch_seq=data_seq(rp(random_batch_size));

    % rp=randperm(60000); 
    % sgd_batch_seq=rp(random_batch_size);

    sgd_selected_images=training.images(:,:,sgd_batch_seq);
    sgd_selected_image_labels=training.labels(sgd_batch_seq);
    A0=fun_image2vector(sgd_selected_images);
    SGD_parameters.A0_labels_in_number=sgd_selected_image_labels;
       
    SGD_Trained_Model=fun_ANN_SGD(W1,b1,W2,b2,A0,SGD_parameters,kk);
    W1=SGD_Trained_Model.W1;
    b1=SGD_Trained_Model.b1;
    W2=SGD_Trained_Model.W2;
    b2=SGD_Trained_Model.b2;

end

save SGD_Trained_Model.mat W1 b1 W2 b2

%%
%%%%%%%%%%%%%%%%%%%%%
% Plot the performance of the SGD-trained model
%

    ANN_parameters.nmb_of_learning_sessions=nmb_of_sessions;

    predicted=fun_ANN(T0,W1,b1,W2,b2);

    figure_number=20;
    fun_plot_input_vs_output(T0,ANN_parameters, predicted.label,figure_number, selected_sample_seq)
    sgtitle("ANN Model: SGD-trained (Sessions = " + nmb_of_sessions + ...
        ", Learning Time per Session = " + nmb_of_searching_per_session + ...
        ", Random Batch Size = " + length(random_batch_size) + ")")
% xshft=2000;
% yshft=-300;
% %%%%%%%laptop setting
% xshft=1800;
% yshft=-308;
% %%%%%%%laptop setting
%%%%%%%laptop setting
xshft=1800;
yshft=-15;
%%%%%%%laptop setting

set(gcf,'Position',[-1800+xshft 295+yshft 1000 290])
set(gcf, 'MenuBar', 'None')
figure(21)
hold off

        ys = SGD_training_positive_rate;
        xs=1:length(ys);
    plot(xs,ys, 'r-','markersize',20)
    if ~isempty(ys)
       axis([-inf xs(end)+10 0.9*ys(1)+1 100])
    end
    title("Positive Rate: " + ys(end) + "")

set(gcf,'Position',[-790+xshft 295+yshft 500 290])
set(gcf, 'MenuBar', 'None')

%%
% Gradient Descent Tunneling 
%
%
predicted=fun_ANN(T0,W1,b1,W2,b2);
profile_output=fun_SGD_Trained_features(ANN_parameters.A0_labels_in_number,ANN_parameters.selected_images_relative_location,predicted.label);

XX=8;
xx=1;
while xx<XX

GDT_training_positive_rate=[];
GDT_training_positive_rate1=[];

% nmb_of_bits=10;
%%%%% for 60000
nmb_of_bits=10;

cloned_position=fun_auxiliary_position(T0_label,...
    profile_output.match_position,predicted.distance,nmb_of_bits);

if ~isempty(profile_output.error_position)
er_lng=length(profile_output.error_position);

[A0_trained, A0_auxiliary, A0_label_in_nmb, A0_label_in_nmb_position]=fun_trained_and_auxiliary_A0(T0, ANN_parameters.A0_labels_in_number,...
    ANN_parameters.A0_labels_in_number_position, profile_output.error_position, profile_output.match_position, cloned_position);


gdi=25; %25
GDT_parameters=ANN_parameters;
GDT_parameters.nmb_of_gradient_decent_iterations=gdi;

bsz=data_size; % 1000 for full 60000 images, grand_batch_size;
match_seq=(er_lng+1):bsz;
er_seq=1:er_lng;
target_seq=[er_seq,match_seq];

GDT_lmd_pr=[];
lmd_pr1=[];

pr0=100;

lmd0=0;

lmd_inc=0.1;
lmd_inc1=lmd_inc;
lmd_cnt_seq_1=0:lmd_inc:1;
lmd_cnt_seq=(lmd0)+(1-lmd0)*lmd_cnt_seq_1;

lmd=lmd_cnt_seq(1);

GDT_lmd_seq=[];
GDT_lmd_seq1=[];
GDT_lmd_inc=[];
GDT_lmd_incbk=[];
GDT_lmd_inc1=[];
GDT_lmd_inc1bk=[];
i=1;
% tic;
cnt_100=6; % continuation steps before step size is doubled. 
W01=W1;
b01=b1;
W02=W2;
b02=b2;
lmd1=[];
j=0;
pr=0;
stopage=0;
adaptive_scaling=3/2;

while lmd<=1 && stopage<2
       %%%%%%%%%%%%%%%%%%%%%%%%
            A0_homotopy=(1-lmd)*A0_auxiliary(:,target_seq)+lmd*A0_trained(:,target_seq);
            GDT_parameters.A0_labels_in_number=A0_label_in_nmb(target_seq);
            GDT_parameters.A0_labels_in_number_position=A0_label_in_nmb_position(target_seq);

            GDT_Trained_Model=fun_ANN_GDT(W1,b1,W2,b2,A0_homotopy,GDT_parameters,lmd,j);
                W1=GDT_Trained_Model.W1;
                b1=GDT_Trained_Model.b1;
                W2=GDT_Trained_Model.W2;
                b2=GDT_Trained_Model.b2;
     
            GDT_lmd_pr=[lmd_pr1,GDT_Trained_Model.positive_rate(1)];
            lmd_pr1=GDT_lmd_pr;
            GDT_lmd_seq=[GDT_lmd_seq1,lmd];
            GDT_lmd_seq1=GDT_lmd_seq;
            GDT_lmd_inc=[GDT_lmd_incbk,lmd_inc];
            GDT_lmd_incbk=GDT_lmd_inc;
            GDT_lmd_inc1=[GDT_lmd_inc1bk,lmd_inc1];
            GDT_lmd_inc1bk=GDT_lmd_inc1;
       %%%%%%%%%%%%%%%%%%%%%%%%

       pr=GDT_Trained_Model.positive_rate(end);
    if pr<pr0
        j=j+1;

        if ~isempty(lmd1)
            i=1;
            W1=W01;
            b1=b01;
            W2=W02;
            b2=b02;

            lmd_inc=lmd_inc1/adaptive_scaling;
            lmd_inc1=lmd_inc;
            lmd00=(lmd1-lmd_inc)*((lmd1-lmd_inc)>0);
            aa=lmd00:lmd_inc:1;
            if mod((1-lmd00),lmd_inc1)>0
               bb=[aa,1];
               lmd_cnt_seq=bb;
            else
               lmd_cnt_seq=aa;
            end           
            lmd=lmd_cnt_seq(i);
            lmd1=lmd;
        else
            lmd=inf;
            disp(['Continuation initials did not hold at iteration: ' num2str(xx)]);
            xx=xx+1;
        end
    end
    if pr>=pr0 && lmd<1
        i=i+1;

        predicted=fun_ANN(T0,W1,b1,W2,b2);
        pp=100*(data_size-sum(1*(abs(T0_label-predicted.label)>0)))/data_size;

        GDT_training_positive_rate=[GDT_training_positive_rate1,pp];
        GDT_training_positive_rate1=GDT_training_positive_rate;

            lmd0=lmd;
            lmd=lmd_cnt_seq(i);
            lmd1=lmd;

        if i==(cnt_100+1)
            lmd_inc1=adaptive_scaling*lmd_inc;
            aa=lmd0:lmd_inc1:1;
            if mod((1-lmd0),lmd_inc1)>0
               bb=[aa,1];
               lmd_cnt_seq=bb;
            else
               lmd_cnt_seq=aa;
            end           

            lmd=lmd_cnt_seq(2);
            lmd1=lmd;
            i=1;
            lmd_inc=lmd_inc1;
        end

            W01=W1;
            b01=b1;
            W02=W2;
            b02=b2;
            j=0;
    end

    if lmd==1 && pr==100
        stopage=stopage+1;
        xx=XX+1;
        GDT_lmd_pr(end)=pr;

        predicted=fun_ANN(T0,W1,b1,W2,b2);
        pp=100*(data_size-sum(1*(abs(T0_label-predicted.label)>0)))/data_size;

        GDT_training_positive_rate=[GDT_training_positive_rate1,pp];
        GDT_training_positive_rate1=GDT_training_positive_rate;

        lmd0=lmd;
        pr=pp;
        GDT_lmd_pr(end)=pr;

    end
end

end
end

W1=GDT_Trained_Model.W1;
b1=GDT_Trained_Model.b1;
W2=GDT_Trained_Model.W2;
b2=GDT_Trained_Model.b2;

save GDT_Trained_Model.mat W1 b1 W2 b2

%%
%%%%%%%%%%%%%%%%%%%%%
% Plot the final performance of the trained model
%
NN_parameters.nmb_of_learning_sessions=nmb_of_sessions;
nn=length(GDT_lmd_pr);

    figure_number=1;

    predicted=fun_ANN(T0,W1,b1,W2,b2);
    fun_plot_input_vs_output(T0,ANN_parameters, predicted.label,figure_number, selected_sample_seq)

    ss=nmb_of_sessions+nn;
    sgtitle("ANN Model: Fully-trained (Tunelling Sessions = " + nn + ")")

% xshft=2000;
% yshft=400;
%%%%%%%laptop setting
xshft=1800;
yshft=275;
%%%%%%%laptop setting

set(gcf,'Position',[-1800+xshft 290+yshft 1000 290])

figure(2)
hold off

tseq=1:length(GDT_training_positive_rate);
xxs=[xs,(tseq)+xs(end)-1];
yys=[ys,GDT_training_positive_rate];
    plot(xxs,yys', 'r-','markersize',20)
    hold on
    plot(xs(end)-1+tseq,GDT_training_positive_rate', 'm-','linewidth',2)
    plot([xs(end),xs(end)], [0 100], 'k--','linewidth',1)

    axis([0,1.0*(tseq(end)+xs(end)), 0 100])
text(1.05*xs(end), 65, {'GDT' 'Region'},'FontSize',14)
text(15, 40, {'SGD' 'Region'},'FontSize',14)

    title("Positive Rate: " + yys(end) + "")
set(gcf,'Position',[-790+xshft 290+yshft 500 290])

%%
figure(3)
hold off

yyaxis left
plot(1:length(GDT_lmd_pr),GDT_lmd_seq,'r-')
ylabel('\lambda_n','color','r','FontSize',14)
axis([-inf inf -inf 1])

yyaxis right
plot(1:length(GDT_lmd_pr),GDT_lmd_inc,'b-')
ylabel('\Delta\lambda_n','color','b','FontSize',14)
axis([-inf inf -inf .2])

xlabel('Tunneling Session','FontSize',14)

ax = gca;
ax.YAxis(1).Color = 'r';
ax.YAxis(2).Color = 'b';

%%
% plot clustering demo
%

plot_clustering_demo

%%