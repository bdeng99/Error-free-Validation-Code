function out=fun_classification_vector2label(vector,label_names)
n=length(label_names); 
v=vector(1:n,:);
I=eye(n);
vv=zeros(n,1);
for i=1:n
vv(i)=(v-I(:,i))'*(v-I(:,i));
end
[aa,idx]=min(vv);
% [~,idx]=max(v);
out.label_words=label_names{idx};
if idx==10
    idx=0;
end
out.label_numbers=idx;
out.label_distance=sqrt(aa);
end