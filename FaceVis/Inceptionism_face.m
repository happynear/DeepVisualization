% Visualize faces in a face recognition model.
% The model can be downloaded from
% https://github.com/AlfredXiangWu/face_verification_experiment .
% This model is trained with CASIA-webface, People_ID can be set from 1
% to 10575, to get the memorized face of the corresponding person.

% Based on paper:
% Feng Wang, Haijun Liu, Jian Cheng, 
% Visualizing Deep Neural Network by Alternately Image Blurring and Deblurring
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

net_model = 'DeepFace_set003_inception.prototxt';% Please, remember to add force_backward:true to this file.
net_weights = 'D:\project\lfw_face_verification_experiment\model\DeepFace_set003_net_iter.caffemodel';
mean_file = [];

train_net = caffe.Net(net_model,net_weights,'test');
if isempty(mean_file)
    mean_image = zeros(128,128);
else
    mean_image = caffe.read_mean(mean_file);
end;
People_ID = 1;

% mean_image = mean_image(17:240,17:240,:);
% mean_image = mean_image + randn(size(mean_image));
input_data = zeros(size(mean_image,1), size(mean_image,2), 1, 1, 'single');

mean_file = 'D:\deepLearning\Visualization\Inceptionism\web_mean.proto';
mean_face = caffe.read_mean(mean_file);
mean_face = mean_face / 256;
input_data(:,:,1,1) = imresize(mean_face,[128,128]);

use_clip = false;
use_cv_norm = false;
use_weight_decay = false;
use_image_blur = false;
use_image_deblur = false;
use_gradient_blur = false;
use_dropout = false;
use_maxGray = false;

H = fspecial('gaussian',[7 7],1.2);
prob = train_net.forward({input_data});
% input_data = input_data - min(input_data(:));
% input_data = input_data / max(input_data(:));
[max_prob,max_idx] = sort(prob{1},'descend');
max_idx = People_ID;%max_idx(1);
this_prob = prob{1}(max_idx);
back_data = ones(size(prob{1}),'single') * -1;
back_data(max_idx) = 1;
back_cell = prob;
back_cell{1} = back_data;
blur_data = zeros(size(input_data));
base_lr = 0.01;
max_lr = 0.01;
lambda1 = 0.00001;
lambda2 = 0.1;
last_prob = -999;
momentum = 0.8;
momentum2 = 0.99;
lastgrad = zeros(size(mean_image));
lastgrad2 = zeros(size(mean_image));
mask = ones(size(mean_image,1), size(mean_image,2));
iter = 1;
dropout = 0.5;

while 1
    lr = base_lr;% * sqrt(this_prob / (1 - this_prob));
    res = train_net.backward(back_cell);
    
    bak_data = input_data;
    
    if use_gradient_blur
        res{1} = imfilter(res{1},H,'same');
    end;
    grad = res{1};
    
    if use_clip
        app_gradient = sum(abs(res{1} .* input_data(:,:,:,1)),3);
        app_gradient = app_gradient < mean(app_gradient(:)) * 0.5;
        clip_grad = reshape(res{1},[size(mean_image,1)*size(mean_image,2) 3]);
        clip_grad(app_gradient==1,:) = 0;
        clip_grad = reshape(clip_grad,size(input_data));
        res{1} = clip_grad;
    end;
    
    if use_cv_norm
        I = input_data(:,:,:,1);
%         Gx = sign(I(2:end-1,2:end-1,:) - I(1:end-2,2:end-1,:)) - sign(I(3:end,2:end-1,:) - I(2:end-1,2:end-1,:));
%         Gy = sign(I(2:end-1,2:end-1,:) - I(2:end-1,1:end-2,:)) - sign(I(2:end-1,3:end,:) - I(2:end-1,2:end-1,:));
        Gx = smoothL1(I(2:end-1,:,:) - I(1:end-2,:,:)) - smoothL1(I(3:end,:,:) - I(2:end-1,:,:));
        Gx = [smoothL1(I(1,:,:) - I(2,:,:)); Gx; smoothL1(I(end,:,:) - I(end-1,:,:))];
        Gy = smoothL1(I(:,2:end-1,:) - I(:,1:end-2,:)) - smoothL1(I(:,3:end,:) - I(:,2:end-1,:));
        Gy = [smoothL1(I(:,1,:) - I(:,2,:)) Gy smoothL1(I(:,end,:) - I(:,end-1,:))];
%         Gx = sign(I(2:end-1,:,:) - I(1:end-2,:,:)) - sign(I(3:end,:,:) - I(2:end-1,:,:));
%         Gx = [sign(I(1,:,:) - I(2,:,:)); Gx; sign(I(end,:,:) - I(end-1,:,:))];
%         Gy = sign(I(:,2:end-1,:) - I(:,1:end-2,:)) - sign(I(:,3:end,:) - I(:,2:end-1,:));
%         Gy = [sign(I(:,1,:) - I(:,2,:)) Gy sign(I(:,end,:) - I(:,end-1,:))];
            grad = grad - lambda2 * (Gx + Gy);
    end;
    
    lastgrad = (1 - momentum) * lr * res{1}   + momentum * lastgrad;%/ norm(res{1}(:))
    input_data(:,:,:,1) = input_data(:,:,:,1) + lastgrad;
%     lastgrad = (1 - momentum) * grad   + momentum * lastgrad;%/ norm(res(:))
%     lastgrad2 = (1 - momentum2) * grad.^2 + momentum2 * lastgrad2;%/ norm(res(:))
%     lg_correct = lastgrad ./ (1 - momentum^iter);
%     lg2_correct = lastgrad2 ./ (1 - momentum2^iter);
%     input_data(:,:,:,1) = input_data(:,:,:,1) + lr * lg_correct ./ (sqrt(lg2_correct) + 1e-8);
    
    
    
    if use_weight_decay
        input_data(:,:,:,1) = input_data(:,:,:,1) - lr * lambda1 * I;
    end;
    if use_maxGray
%         if max(input_data(:))>1
            input_data(:,:,:,1) = input_data(:,:,:,1) - min(input_data(:));
            input_data(:,:,:,1) = input_data(:,:,:,1) / max(input_data(:));
%         end;
    end;
%         input_data = (input_data -mean(input_data(:))) / std(input_data(:)) * 30;
%     end;

%     for_forward = reshape(input_data,[size(mean_image,1)*size(mean_image,2) 3]);
%     mask = rand(size(mean_image,1), size(mean_image,2)) < dropout;
%     for_forward(mask==1,:) = 0;
%     for_forward = reshape(for_forward,size(input_data));
    
    if mod(iter,10) ==0%&&iter<2000
        if mod(iter,20) ~= 0
            H = fspecial('gaussian',[5 5],rand()/2+0.5);
            if use_image_blur
                input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
            end
        else
            if mod(iter,20) == 0
                if use_image_deblur
                    input_data(:,:,:,1) = deconvlucy(input_data(:,:,:,1), H);
                end;
            end;
        end;
    end;
    prob = train_net.forward({input_data});
    
    this_prob = prob{1}(max_idx);
    fprintf('iter=%d,lr=%f,prob1=%f,last_prob=%f\n',iter,lr,prob{1}(max_idx),last_prob);
    iter = iter + 1;
    
    if mod(iter,100)==0
        figure(2);
        % imshow(uint8(mean_image + input_data));
        output = input_data(:,:,:,1);
        output = output';
        imshow(output);
        I = output;
%         Gx = abs(I(2:end-1,2:end-1,:) - I(1:end-2,2:end-1,:)) + abs(I(3:end,2:end-1,:) - I(2:end-1,2:end-1,:));
%         Gy = abs(I(2:end-1,2:end-1,:) - I(2:end-1,1:end-2,:)) + abs(I(2:end-1,3:end,:) - I(2:end-1,2:end-1,:));
%         figure(3);hist(I(:),1000);
%         figure(4);hist(Gy(:),1000);
        if iter == 200
            break;
        end;
    end;
    if this_prob<last_prob
        base_lr = base_lr * 0.99;
%         input_data = bak_data;
    end;
    if this_prob>last_prob&&base_lr<max_lr%&& (this_prob-last_prob) / this_prob < 0.001
        base_lr = base_lr * 1.01;
%         input_data = bak_data;
    end;
%     if this_prob>last_prob
        last_prob = this_prob;
%     end;
    if lr<0.000001
        break;
    end;
end;
figure(2);
% imshow(uint8(mean_image + input_data));
output = input_data(:,:,:,1);
output = output';
imshow(output);