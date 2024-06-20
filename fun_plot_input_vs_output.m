function []=fun_plot_input_vs_output(A0,parameters, predicted_labels,figure_number, sample_sequence)
%
%  
% sample_sequence=selected_sample_seq;
fs=16;
ms=16;
out=fun_classification_vector2image_and_label(A0,parameters.image_size,parameters.A0_labels_in_number');

mtg=out.images(:,:,sample_sequence);
mtglb=out.labels(sample_sequence);

% nmb_of_images=length(A0(1,:));
nmb_of_images=length(sample_sequence);

rn=floor(sqrt(nmb_of_images));
rownmb=max([min([10,rn]),1]);
colnmb=rn;

figure(figure_number)
hold off
% montage(rescale(mtg,0,2^8));
subplot(1,3,1)
hold off
montage(mtg,"Size",[rownmb colnmb],"DisplayRange",[0 1], "BorderSize", [4 4],"BackgroundColor","w"); %,"BackgroundColor","w"
% montage(mtg,"Size",[10 10]); 
title('Images')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pmtglb=predicted_labels(sample_sequence);
aa=abs(mtglb'-pmtglb);

subplot(1,3,2)
hold off
plot(1,1)
hold on
for j=1:rownmb
    for i=1:colnmb
text(i,rownmb+1-j,num2str(mtglb((j-1)*rownmb+i)),'FontSize',fs)
        if aa((j-1)*rownmb+i)>0
            plot(i+.1,rownmb+1-j-.05, 'bo','markersize',ms)
%         text(i,rownmb+1-j,num2str(pmtglb((j-1)*rownmb+i)),'color','c')
        end
    end
end
title('True Labels')
axis([0 colnmb+1 0 rownmb+1])
h=gca; 
h.XAxis.TickLength = [0 0];
h.YAxis.TickLength = [0 0];
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
axis square

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulated output
% sq=randperm(length(mtglb));
% mtglb=mtglb(sq);
pmtglb=predicted_labels(sample_sequence);
% Enter true output here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

aa=abs(mtglb'-pmtglb);
% idx=(abs(mtglb-pmtglb)>0);
% iidx=1*idx;
% % errpmtglb=pmtglb(idx);

subplot(1,3,3)
hold off
plot(1,1)
hold on
for j=1:rownmb
    for i=1:colnmb
text(i,rownmb+1-j,num2str(pmtglb((j-1)*rownmb+i)),'FontSize',fs)
        if aa((j-1)*rownmb+i)>0
            plot(i+.1,rownmb+1-j-.05, 'ro','markersize',ms)
        text(i,rownmb+1-j,num2str(pmtglb((j-1)*rownmb+i)),'color','r','FontSize',fs)
        end
    end
end
title('Predicted Labels')
axis([0 colnmb+1 0 rownmb+1])
h=gca; 
h.XAxis.TickLength = [0 0];
h.YAxis.TickLength = [0 0];
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
axis square

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sgtitle("FNN Model: Input vs Output (Training time = " + parameters.nmb_of_learning_sessions(end) + ")")


end