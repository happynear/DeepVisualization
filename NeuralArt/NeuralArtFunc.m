%% Neural Art with lbfgs optimization
%% It cannot run on my laptop T^T. The error seems to be memory out.
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% original_file = 'D:\deepLearning\caffe-windows\matlab\demo\Inceptionism\googlenet_neuralart.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_googlenet.caffemodel';
% style_layer = {'icp2_in','icp3_out','icp5_out','icp7_out','icp9_out'};
original_file = 'D:\deepLearning\caffe-windows\matlab\demo\NeuralArt\VGG_16_nueralart.prototxt';
net_weights = 'D:\deepLearning\caffe-windows\matlab\demo\NeuralArt\VGG16_thinned_net.caffemodel';
style_layer = {'conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'};
content_layer = {'conv4_2'};

style_weights = [1 1 1 1 1 0.05];
tv_norm = 0.001;
use_color_prior = false;
num_cluster=6;
color_prior = 0.5;
long_size = 512;

style_image = imread('d:\starry_night.jpg');
if size(style_image,1) > size(style_image,2)
    style_image = imresize(style_image,[long_size, size(style_image,2) / size(style_image,1) * long_size]);
else
    style_image = imresize(style_image,[size(style_image,1) / size(style_image,2) * long_size, long_size]);
end;
content_image = imread('d:\hoovertowernight.jpg');
if size(content_image,1) > size(content_image,2)
    content_image = imresize(content_image,[long_size, size(content_image,2) / size(content_image,1) * long_size]);
else
    content_image = imresize(content_image,[size(content_image,1) / size(content_image,2) * long_size, long_size]);
end;
figure(1);
imshow(style_image);
title('style image');
figure(2);
imshow(content_image);
title('content image');
[style_generate_prototxt, style_pattern, content_pattern,colorObj] = MakeStylePrototxt(original_file, net_weights, style_layer, style_weights, content_layer, style_image, content_image,num_cluster);
if use_color_prior
colorObj
num_cluster = colorObj.NComponents;
end;
if use_color_prior
    gaussian_net = caffe.Net('gaussian_net.prototxt','test');
    
    W = single(zeros(1,1,3,3*num_cluster));
    b = single(zeros(3*num_cluster,1));

    for i=1:num_cluster
        W(:,:,:,(i-1)*3+1:i*3) = inv(colorObj.Sigma(:,:,i));
        b((i-1)*3+1:i*3) = -1 * colorObj.Sigma(:,:,i) \ colorObj.mu(i,:)';
    end;

    nth_layer = gaussian_net.layer_vec(gaussian_net.name2layer_index('gaussian_prior'));
    nth_layer.params(1).set_data(W);
    nth_layer.params(2).set_data(b);
end;

forward_input = cell(length(style_pattern)+2,1);
forward_input(3:end) = style_pattern;
forward_input(2) = {content_pattern};

mean_file = [];
vgg_mean =  [103.939, 116.779, 123.68];
[height, width, channel] = size(content_image);

%%%%%%%%%extract the train features
stylegen_net = caffe.Net(style_generate_prototxt,net_weights,'test');
mean_image = permute(repmat(vgg_mean',[1,size(content_image,2),size(content_image,1)]),[2,3,1]);
input_data = randn(size(mean_image,1), size(mean_image,2), 3, 1, 'single')*50;

lr = 10;
max_lr = 50;
momentum = 0.8;
lastgrad = zeros(size(mean_image));
last_cost = 9999999999999;

for iter = 1:2000
    bak_data = input_data;
    bak_grad = lastgrad;
    [cost, grad] = NeuralArtCost(input_data,[],stylegen_net,forward_input,style_weights);
    
    if tv_norm > 0
        I = input_data(:,:,:,1);
        Gx = (I(2:end-1,:,:) - I(1:end-2,:,:)) - (I(3:end,:,:) - I(2:end-1,:,:));
        Gx = [(I(1,:,:) - I(2,:,:)); Gx; (I(end,:,:) - I(end-1,:,:))];
        Gy = (I(:,2:end-1,:) - I(:,1:end-2,:)) - (I(:,3:end,:) - I(:,2:end-1,:));
        Gy = [(I(:,1,:) - I(:,2,:)) Gy (I(:,end,:) - I(:,end-1,:))];
        grad = grad + tv_norm * (Gx+Gy);
    end;
    
    if use_color_prior
        gmm_prior = gaussian_net.forward({input_data});
        sum_gp = zeros(size(mean_image,1),size(mean_image,2));
        sum_prob_gradient = zeros(size(mean_image));
        for i=1:num_cluster
            gp = bsxfun(@minus,input_data(:,:,:,1),reshape(colorObj.mu(i,:),[1 1 3])) .* gmm_prior{1}(:,:,(i-1)*3+1:i*3);
            gp = sum(gp,3);
            gp = colorObj.PComponents(i) * exp(-gp);
            sum_prob_gradient = sum_prob_gradient + bsxfun(@times,gp,gmm_prior{1}(:,:,(i-1)*3+1:i*3));
            sum_gp = sum_gp + gp;
        end;
        sum_prob_gradient = bsxfun(@rdivide,sum_prob_gradient,sum_gp);
        sum_prob_gradient(isnan(sum_prob_gradient)) = 0;
        input_data(:,:,:,1) = input_data(:,:,:,1) - lr * color_prior * sum_prob_gradient;
    end;
    %%%%%%%%%%%%%%%%%%%%%%%%gd linear search
    lastgrad = (1 - momentum) * lr * grad   + momentum * lastgrad;%/ norm(res(:))
    input_data(:,:,:,1) = input_data(:,:,:,1) - lastgrad;
    if cost>last_cost
        lr = lr * 0.9;
%         input_data = bak_data;%why...
%         last_grad = bak_grad;
    end;
    if cost<last_cost&&lr<max_lr%&& (this_prob-last_prob) / this_prob < 0.001
        lr = lr * 1.1;
    end;
    if cost<last_cost
        last_cost = cost;
    end;
    fprintf('iter=%d,lr=%f,this_cost=%f,last_cost=%f\n',iter,lr,cost, last_cost);
    %%%%%%%%%%%%%%%%%%%%%%gd
    
    if mod(iter,10)==0
        figure(3);
        output = mean_image + input_data(:,:,:,1);
        output = output(:, :, [3, 2, 1]);
        output = permute(output, [2 1 3]);
        imshow(uint8(output));
        I = output;
        title('generated image');
    end;
end;