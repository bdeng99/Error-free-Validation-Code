function out=fun_SGD_Trained_features(true_label_in_number, true_label_location, predicted_labels)
%
%  
%

predicted_label_in_number=predicted_labels;

ln=length(true_label_in_number);
tl=true_label_in_number;
pl=predicted_label_in_number;

idx=(abs(tl-pl)>0);
out.error_position=true_label_location(idx);
err_positions=1*idx;

idx=(err_positions==0);
out.match_position=true_label_location(idx);

idx=(err_positions>0);
pr=100*(ln-sum(1*idx))/ln;
out.positive_rate=pr;

end