function [A0_trained, A0_auxiliary, A0_label_in_nmb, A0_label_in_nmb_position]=fun_trained_and_auxiliary_A0(T0,true_image_labels, ...
    image_label_number, error_position, match_position, cloned_position)
%
%
%

n=length(cloned_position(:,1));
match_position_1=match_position;
%%%%%
A0_match=T0(:,match_position_1);
A0_label_in_nmb_match_0=true_image_labels(match_position_1);
A0_label_in_nmb_position_match_0=image_label_number(match_position_1);

A0_trained_0=A0_match;
A0_auxiliary_0=A0_match;

aa=true_image_labels;
bb=error_position;
cc=aa(bb);
    for kk=0:9
        idx=(cc==kk);
        dd=bb(idx);
        if ~isempty(dd)
            for zz=1:length(dd)
                A0_trained=[T0(:,dd(zz)), A0_trained_0];
                A0_trained_0=A0_trained;
                A0_auxiliary=[T0(:,cloned_position(randi([1 n],1),kk+1)), A0_auxiliary_0];
                A0_auxiliary_0=A0_auxiliary;
                A0_label_in_nmb_match=[true_image_labels(dd(zz)),A0_label_in_nmb_match_0];
                A0_label_in_nmb_match_0=A0_label_in_nmb_match;
                A0_label_in_nmb_position_match=[image_label_number(dd(zz)),A0_label_in_nmb_position_match_0];
                A0_label_in_nmb_position_match_0=A0_label_in_nmb_position_match;
            end
        end
    end
A0_label_in_nmb=A0_label_in_nmb_match_0';  
A0_label_in_nmb_position=A0_label_in_nmb_position_match_0;  
end