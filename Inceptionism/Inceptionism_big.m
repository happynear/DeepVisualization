% Visualization of the classifier neurons. Modify the targetclass variable
% to see other classes.
% Please feel free to modify all the switches in line 40 - 46 and width,
% height, scale, rotate variables to see what happens.
% The best visualization can be produced by setting use_image_blur = true
% and use_image_deblur = true, with all other switches are turned to false.
% However, with no constraints, you can still get recognizable images.
% Feel the magic by your self!
%
% This code is messay, I will refactor it in the future.
% Based on paper:
% Feng Wang, Haijun Liu, Jian Cheng, 
% Visualizing Deep Neural Network by Alternately Image Blurring and Deblurring
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% net_model = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\googlenet_deploy_prob3.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_googlenet.caffemodel';
net_model ='D:\deepLearning\caffe-windows\models\bvlc_googlenet\deploy.prototxt';
net_weights = 'D:\deepLearning\caffe-windows\models\bvlc_googlenet\thinned.caffemodel';
% mean_file = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_mean.binaryproto';

% net_model = 'D:\deeplearning\caffe-windows\examples\PlaceCNN\hybridCNN_deploy_upgraded.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\PlaceCNN\hybridCNN_iter_700000_upgraded.caffemodel';
% mean_file = 'D:\deeplearning\caffe-windows\examples\PlaceCNN\hybridCNN_mean.binaryproto';

% net_model = 'D:\deeplearning\caffe-windows\examples\VGG\VGG_ILSVRC_16_layers_deploy.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\VGG\VGG_ILSVRC_16_layers.caffemodel';
% mean_file = [];
% vgg_mean =  [103.939, 116.779, 123.68];
% net_model = 'D:\deeplearning\caffe-windows\models\bvlc_googlenet\deploy.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\models\bvlc_googlenet\bvlc_googlenet.caffemodel';
mean_file = [];
vgg_mean =  [103.939, 116.779, 123.68];

%%%%%%%%%extract the train features
train_net = caffe.Net(net_model,net_weights,'test');
use_clip = false;
use_cv_norm = false;
use_weight_decay = false;
use_image_blur = true;
use_image_deblur = true;
use_gradient_blur = false;
use_dropout = false;

for targetclass = 13:13
%     if exist(['gallery\' num2str(targetclass) '.png'],'file')
%         continue;
%     end;
if ~isempty(mean_file)
    mean_image = caffe.read_mean(mean_file);
    mean_image = mean_image(16:242,16:242,:);
else
    mean_image = permute(repmat(vgg_mean',[1,224,224]),[2,3,1]);
end;
% mean_image = mean_image + randn(size(mean_image));
input_data = zeros(size(mean_image,1), size(mean_image,2), 3, 1, 'single');

width = 300;
height = 300;
meanV = mean(mean_image(:));
BigImage = permute(repmat(vgg_mean',[1,height,width]),[2,3,1]);%this image will be transposed
H = fspecial('gaussian',[7 7],1.2);

if length(targetclass) > 1
    sample_density = 2;
    class_center = zeros(sample_density*length(targetclass),2);
    for i = 1:sample_density*length(targetclass)
        class_center(i,1) = randi(height - size(mean_image,1),1,1) + size(mean_image,1) / 2;
        class_center(i,2) = randi(width - size(mean_image,2),1,1) + size(mean_image,2) / 2;
        if i > 1
            min_distance = min(pdist2(class_center(i,:),class_center(1:i-1,:)));
            while min_distance < size(mean_image,1) / 2
                class_center(i,1) = randi(height - size(mean_image,1),1,1) + size(mean_image,1) / 2;
                class_center(i,2) = randi(width - size(mean_image,2),1,1) + size(mean_image,2) / 2;
                min_distance = min(pdist2(class_center(i,:),class_center(1:i-1,:)));
            end;
        end;
    end;
    
    class_label = repmat(targetclass,1,sample_density);
end;

blur_data = zeros(size(input_data));
base_lr = 5000;
max_lr = 5000;
lambda1 = 0.01;
lambda2 = 0.1;
lambda3 = 300;
last_prob = -999;
momentum = 0.8;
lastgrad = zeros(size(mean_image));
mask = ones(size(mean_image,1), size(mean_image,2));
iter = 1;
dropout = 0.5;
sep_iter = zeros(ceil((height - size(mean_image,1)) / size(mean_image,1) *2),ceil((width - size(mean_image,2)) / size(mean_image,2) * 2));
erode = 5;
start_sigma = 1.2;
end_sigma = 0.5;
this_sigma = start_sigma;
sigma_decay = 0.99;
deblur_sigma_ratio = 0.8;
scale = [0.8 1.2];
rotate = [0 0] / 180 * pi;
func_loss = [];
BigGrad = zeros(size(BigImage));

while 1
%     for i = 0:size(mean_image,1) / 2:height - size(mean_image,1)
%         for j = 0:size(mean_image,2) / 2:width - size(mean_image,2)
            this_sigma = this_sigma * sigma_decay;
            if this_sigma < end_sigma
                this_sigma = end_sigma;
            end;
            s = rand(1) * (scale(2)-scale(1)) + scale(1);
            r = rand(1) * (rotate(2)-rotate(1)) + rotate(1);
            if length(targetclass)>1 && iter < 500
                idx = randi(size(class_center,1),1,1);
                x = class_center(idx,1) - size(mean_image,1) / 2;
                y = class_center(idx,2) - size(mean_image,2) / 2;
            else
                if rand() > size(mean_image,1) / height
                    x = randi(height - floor(size(mean_image,1) * s),1,1);% + size(mean_image,1) / 2;
                else
                    pos = randi(2,1,1)-1;
                    x = pos*(height - floor(size(mean_image,1) * s));
                end;
                if rand() > size(mean_image,2) / width
                    y = randi(width - floor(size(mean_image,2) * s),1,1);% - size(mean_image,2) / 2;
                else
                    pos = randi(2,1,1)-1;
                    y = pos*(width - floor(size(mean_image,2) * s));
                end;
            end;
            
            II = zeros(floor(size(mean_image,1) * s),floor(size(mean_image,2) * s),3);
            II(:,:,1) = BigImage(x + 1:x + floor(size(mean_image,1) * s),y + 1:y + floor(size(mean_image,2) * s),1) - vgg_mean(1);
            II(:,:,2) = BigImage(x + 1:x + floor(size(mean_image,1) * s),y + 1:y + floor(size(mean_image,2) * s),2) - vgg_mean(2);
            II(:,:,3) = BigImage(x + 1:x + floor(size(mean_image,1) * s),y + 1:y + floor(size(mean_image,2) * s),3) - vgg_mean(3);
            input_data(:,:,:,1) = imresize(imrotate(II,r),[size(mean_image,1), size(mean_image,2)]);
            GG = BigGrad(x + 1:x + floor(size(mean_image,1) * s),y + 1:y + floor(size(mean_image,2) * s),:);
            last_grad = imresize(GG,[size(mean_image,1), size(mean_image,2)]);
            for k = 1 : 10
                
                lr = base_lr;% * sqrt(this_prob / (1 - this_prob));
                
                prob = train_net.forward({input_data});
                
                    this_prob = prob{end}(targetclass);
                [max_prob,max_idx] = max(this_prob);
                fprintf('iter=%d,lr=%f,max_idx=%d,prob1=%f,last_prob=%f,this_sigma=%f\n',iter,lr,max_idx,this_prob(max_idx),last_prob,this_sigma);
                func_loss = [func_loss; this_prob(max_idx)];
                
                if k==1
                    back_data = ones(size(prob{1}),'single') * -1;
%                     back_data = zeros(size(prob{end}),'single');
                    if length(targetclass) >1
                        center_x = x + size(mean_image,1) / 2;
                        center_y = y + size(mean_image,2) / 2;
                        distances = pdist2([center_x center_y],class_center);
                        distances = exp(-(distances / size(mean_image,1) * 4).^2);
                        distances = distances / sum(distances);
                        distances = reshape(distances,length(targetclass),sample_density);
                        distances = sum(distances,2);
                        distances = distances * 2 - 1;
                        back_data(targetclass) = distances';
                    else
                        back_data(targetclass(max_idx)) = 1;
                    end;

                    back_cell = cell(length(prob),1);
%                     back_cell{1} = zeros(size(back_data));
%                     back_cell{2} = zeros(size(back_data));
%                     back_cell{1} = back_data;
%                     back_cell{2} = back_data;
                    back_cell{1} = back_data;
                end;
                
                iter = iter + 1;
                
%                 sep_iter(floor(i / size(mean_image,1) * 2)+1,floor(j/size(mean_image,1)*2) + 1) = sep_iter(floor(i / size(mean_image,1) * 2)+1,floor(j/size(mean_image,1)*2) + 1) + 1;
                
                if max_prob<last_prob
                    base_lr = base_lr * 0.99;
            %         input_data = bak_data;
                end;
                if max_prob>last_prob&&base_lr<max_lr%&& (this_prob-last_prob) / this_prob < 0.001
                    base_lr = base_lr * 1.01;
            %         input_data = bak_data;
                end;
            %     if this_prob>last_prob
                    last_prob = max_prob;
            %     end;
            
                res = train_net.backward(back_cell);

                bak_data = input_data;

                if use_clip
                    app_gradient = sum(abs(res{1} .* input_data(:,:,:,1)),3);
                    app_gradient = app_gradient < mean(app_gradient(:));
                    grad = reshape(res{1},[size(mean_image,1)*size(mean_image,2) 3]);
                    grad(app_gradient==1,:) = 0;
                    grad = reshape(grad,size(input_data));
                    res{1} = grad;
                end;

                input_data(:,:,:,1) = input_data(:,:,:,1) + res{1} * 1.5 / mean(abs(res{1}(:)));

                if use_cv_norm
                    I = input_data(:,:,:,1);
                    Gx = smoothL1(I(2:end-1,:,:) - I(1:end-2,:,:)) - smoothL1(I(3:end,:,:) - I(2:end-1,:,:));
                    Gx = [smoothL1(I(1,:,:) - I(2,:,:)); Gx; smoothL1(I(end,:,:) - I(end-1,:,:))];
                    Gy = smoothL1(I(:,2:end-1,:) - I(:,1:end-2,:)) - smoothL1(I(:,3:end,:) - I(:,2:end-1,:));
                    Gy = [smoothL1(I(:,1,:) - I(:,2,:)) Gy smoothL1(I(:,end,:) - I(:,end-1,:))];
                    input_data(:,:,:,1) = input_data(:,:,:,1) - lr * lambda2 * (Gx + Gy);
                end;
                if use_weight_decay
                    input_data(:,:,:,1) = input_data(:,:,:,1) - lambda1 * input_data(:,:,:,1);
                end;
                
                if k==1
                    H = fspecial('gaussian',[randi(6)+4 7],this_sigma);
                    if use_image_blur
                        input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
                    end
                end;
                if k==6
                    H = fspecial('gaussian',[randi(6)+4 7],this_sigma * deblur_sigma_ratio);
                    if use_image_deblur
                        input_data(:,:,:,1) = deconvlucy(input_data(:,:,:,1), H);
                    end;
                end;
                
                if lr<0.000001
                    break;
                end;
            end;
            II = imresize(imrotate(input_data(:,:,:,1),-r),floor([size(mean_image,1), size(mean_image,2)] * s));
            BigImage(x + 1+erode:x + floor(size(mean_image,1) * s)-erode,y + 1+erode:y + floor(size(mean_image,2) * s)-erode,1) = II(erode+1:end-erode,erode+1:end-erode,1) + vgg_mean(1);
            BigImage(x + 1+erode:x + floor(size(mean_image,1) * s)-erode,y + 1+erode:y + floor(size(mean_image,2) * s)-erode,2) = II(erode+1:end-erode,erode+1:end-erode,2) + vgg_mean(2);
            BigImage(x + 1+erode:x + floor(size(mean_image,1) * s)-erode,y + 1+erode:y + floor(size(mean_image,2) * s)-erode,3) = II(erode+1:end-erode,erode+1:end-erode,3) + vgg_mean(3);
            GG = imresize(last_grad,floor([size(mean_image,1), size(mean_image,2)] * s));
            BigGrad(x + 1+erode:x + floor(size(mean_image,1) * s)-erode,y + 1+erode:y + floor(size(mean_image,2) * s)-erode,:) = GG(erode+1:end-erode,erode+1:end-erode,:);
%             sort_data = sort(BigImage(:));
%             BigImage = BigImage .* (255 / sort_data(floor(length(sort_data) * 0.999)));
            if mod(iter,100)==1
                figure(2);
                % imshow(uint8(mean_image + input_data));
                output = BigImage;
                output = output(:, :, [3, 2, 1]);
                output = permute(output, [2 1 3]);
                output = output(erode+1:end-erode,erode+1:end-erode,:);
                sort_output = sort(output(:));
                output = output .* (255 / sort_output(floor(length(sort_output) * 0.99)));
                imshow(uint8(output));
                figure(3);
%                 hist(output(:),1000);
                plot(1:iter-1, func_loss);
            end;
            if iter > 1500
                break;
            end;
%         end;
%     end;
end;
LastBigImage = BigImage;
imwrite(uint8(output),['gallery\' num2str(targetclass) '.png']);
end;