function image_category_vector=fun_classification_label2vector(image_labels,num_of_labels)
I=eye(num_of_labels);
image_category_vector=I(:,image_labels);
end