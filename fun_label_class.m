function out=fun_label_class(label,selected_image_labels,nmb_of_cluster_label)
%
% nmb_of_cluster_label=10;

aa=selected_image_labels;
lb_pst=1:length(aa);
idx=(aa==label);
bb=lb_pst(idx);
n=length(bb);
out=bb(1:min(n,nmb_of_cluster_label));
end
