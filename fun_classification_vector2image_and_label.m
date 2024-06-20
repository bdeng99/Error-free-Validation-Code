function out=fun_classification_vector2image_and_label(A0,image_size,image_labels)
%
%
k=length(A0(1,:));
im=zeros(image_size(1),image_size(2),k);
for i=1:k
    im(:,:,i)=reshape(A0(:,i),image_size);
end
out.images=im;
out.image_size=image_size;
out.labels=image_labels;
end