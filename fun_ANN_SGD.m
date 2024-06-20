function SGD_Trained_Model=fun_ANN_SGD(W1,b1,W2,b2,A0,SGD_parameters,kk)
%
%
nmb_of_hidden_nodes=length(b1);
nmb_of_labels=length(b2);
N=length(A0(1,:));

image_labels=SGD_parameters.A0_labels_in_number;
idx=(image_labels==0);
image_labels(idx)=10;
training_vectors=fun_classification_label2vector(image_labels,nmb_of_labels);
T=SGD_parameters.nmb_of_gradient_decent_iterations;
    
aaa=zeros(T,1)+NaN; % same as positive_rate(:,1);

    for gradient_decent_iteration=1:T 
                       
       gradient_norm_squared=0;

    %%%%% initializing activation and classification variables
       A1=zeros(nmb_of_hidden_nodes,N);
       dA1=zeros(nmb_of_hidden_nodes,nmb_of_hidden_nodes,N);
       A2=zeros(nmb_of_labels,N);
       dA2=zeros(nmb_of_labels,nmb_of_labels,N);
    %%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%

    for j=1:N
        z1=W1*A0(:,j)+b1;
        [a1,da1]=fun_activation(z1);
         A1(:,j)=a1;
         dA1(:,:,j)=da1;
    
        z2=W2*a1+b2;
        [a2,da2]=fun_prediction(z2);

         A2(:,j)=a2;
         dA2(:,:,j)=da2;
    end

%%%%
%%%% backpropagation to find new gradients in W, b
%%%%                  page-wise operations
%%%%
              phi_minus_c=(A2-training_vectors)/N;
            
              dLdZ=fun_star_multiplication(phi_minus_c', dA2);
                gradient_b2=sum((dLdZ'),2);
                gradient_W2=(A1*dLdZ)';
                gradient_norm_squared=gradient_norm_squared+fun_norm_squared(gradient_b2)+fun_norm_squared(gradient_W2);

             dLdZ=fun_star_multiplication(dLdZ*W2, dA1);
                gradient_b1=sum((dLdZ'),2);
                gradient_W1=(A0*dLdZ)';
                gradient_norm_squared=gradient_norm_squared+fun_norm_squared(gradient_b1)+fun_norm_squared(gradient_W1);

    %%%%%%%%%%%% create learning rate
                nrm=sqrt(gradient_norm_squared);
                learning_rate=1.0e-2*(0.2e+2*exp(-nrm)/(1+exp(-nrm))+1)/nrm; % adaptive rate
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            predicted=fun_ANN(A0,W1,b1,W2,b2);
            pr=100*(N-sum(1*(abs(SGD_parameters.A0_labels_in_number'-predicted.label)>0)))/N;
            aaa(gradient_decent_iteration,1)=pr;
    %%%%%%%%%%%% gradient descent update
                    W1=W1-learning_rate*gradient_W1;
                    b1=b1-learning_rate*gradient_b1;           
                    W2=W2-learning_rate*gradient_W2;
                    b2=b2-learning_rate*gradient_b2;           
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% plot intermediate result %%%%%%%%%%%%%%%%
    wk_parameters=SGD_parameters;
    wk_parameters.nmb_of_gradient_decent_iterations=gradient_decent_iteration;
    wk_parameters.nmb_of_learning_sessions=gradient_decent_iteration;
    wk_parameters.image_size=[28,28];

         if gradient_decent_iteration==1 || mod(gradient_decent_iteration,fix(T/3))==0
 
            predicted=fun_ANN(A0,W1,b1,W2,b2);
            nmb_of_plot_image=min(wk_parameters.display_number, N);
            selected_sample_seq=1:nmb_of_plot_image;
            figure_number=1;

            fun_plot_input_vs_output(A0,wk_parameters, predicted.label',figure_number, selected_sample_seq)

            sgtitle("ANN Model: SGD Training (Session = " + kk + ...
            ", Learning Time = " + SGD_parameters.nmb_of_gradient_decent_iterations + ...
            ", Random Batch Size = " + N + ")")

%%%%%%%laptop setting
xshft=1800;
yshft=-15;
%%%%%%%laptop setting

            set(gcf,'Position',[-1800+xshft 300+yshft 1000 290])
            set(gcf, 'MenuBar', 'None')

            figure(figure_number+1)
            hold off

            xs=1:length(aaa);
            xs = xs(~isnan(aaa));
            ys = aaa(~isnan(aaa));
            plot(xs,ys', 'r-','markersize',20,'LineWidth',1.5)
            if ~isempty(ys)
               axis([-inf xs(end)+10 0.9*ys(1)+1 100])
            end
            title("Positive Rate: " + ys(end) + "")
set(gcf,'Position',[-790+xshft 300+yshft 500 290])
set(gcf, 'MenuBar', 'None')
 
            figure(figure_number)

        end

    end

    SGD_Trained_Model.W1=W1;
    SGD_Trained_Model.b1=b1;
    SGD_Trained_Model.W2=W2;
    SGD_Trained_Model.b2=b2;

%%%%%%%%%%%%%%%%% 
    function nrm=fun_norm_squared(B)
        C=B.*B;
        nrm=sum(sum(C,1),2);
    end
%%%%%%%%%%%%%%%%%
end