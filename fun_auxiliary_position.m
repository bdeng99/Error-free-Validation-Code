function auxiliary_position=fun_auxiliary_position(true_image_labels, true_match_position, A_end_classification_distance,nmb_of_bits)
%
%
%
% true_image_labels=T0_label;
% true_match_position=profile_output.match_position;
% A_end_classification_distance=predicted.distance;

n=nmb_of_bits;
helper=zeros(n,10);
% aa=selected_image_labels';
aa=true_image_labels;
bb=true_match_position;
cc=aa(bb);
for ii=0:9
    idx=(cc==ii);
    dd=bb(idx);
    if ~isempty(dd)    
    [~,idx]=sort(A_end_classification_distance(dd));
    ee=dd(idx);
    if length(ee)<n
        eee=ones(n,1)*ee;
        eee=eee';
        ee=eee(:)';
         % disp('Not enough fulled trained label for auxiliary data. Start it over.');
         % return
    end
    helper(:,ii+1)=ee(1:n)';
    else
         disp('Missing fully trained label to complete GDT. Start it over.');
         return
    end
end
auxiliary_position=helper;
end