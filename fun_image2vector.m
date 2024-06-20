function image_vector=fun_image2vector(image)
%
%
%
    nmb_of_images=length(image(1,1,:));
    image_size=size(image(:,:,1));
    vct_lng=image_size(1)*image_size(2);
    image_vector=zeros(vct_lng,nmb_of_images);
    for k=1:nmb_of_images
        aa=image(:,:,k);
        image_vector(:,k)=aa(:);
    end
end