%% Neural Art with lbfgs optimization
%% It cannot run on my laptop T^T. The error seems to be memory out.
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% original_file = 'D:\deepLearning\caffe-windows\matlab\demo\Inceptionism\googlenet_neuralart.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_googlenet.caffemodel';
original_file = 'D:\deepLearning\caffe-windows\matlab\demo\NeuralArt\VGG_16_nueralart.prototxt';
net_weights = 'D:\deepLearning\caffe-windows\matlab\demo\NeuralArt\VGG16_thinned_net.caffemodel';
% style_layer = {'icp2_in','icp3_out','icp5_out','icp7_out','icp9_out'};
style_layer = {'conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'};
style_weights = [1 1 1 1 1 0.1];
content_layer = {'conv4_2'};
style_image = imread('d:\starry_night.jpg');
content_image = imread('d:\tubingen.jpg');
content_image = imresize(content_image,0.25,'bilinear');
% content_image = content_image(end-199:end,1:200,:);
[style_generate_prototxt, style_pattern, content_pattern] = MakeStylePrototxt(original_file, net_weights, style_layer, style_weights, content_layer, style_image, content_image);

forward_input = cell(length(style_pattern)+2,1);
forward_input(3:end) = style_pattern;
forward_input(2) = {content_pattern};

mean_file = [];
vgg_mean =  [103.939, 116.779, 123.68];
[height, width, channel] = size(content_image);

%%%%%%%%%extract the train features
stylegen_net = caffe.Net(style_generate_prototxt,net_weights,'test');
mean_image = permute(repmat(vgg_mean',[1,size(content_image,2),size(content_image,1)]),[2,3,1]);
im_data = content_image(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = reshape(im_data,[width, height, 3, 1]);
for c = 1:3
    im_data(:, :, c, :) = im_data(:, :, c, :) - vgg_mean(c);
end
input_data = randn(size(mean_image,1), size(mean_image,2), 3, 1, 'single')*50;
% input_data = im_data;
% H = fspecial('gaussian',[17 17],5);
% input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');

use_clip = false;
use_cv_norm = true;
cv_norm = 0.001;

% Use minFunc to minimize the function
addpath minFunc/minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';
options.logfile = 'lbfgs.log';

[optData, cost] = minFunc( @(p) NeuralArtCost(p, ...
                                   size(input_data),stylegen_net, ...
                               forward_input,style_weights,cv_norm), ...
                               input_data(:),options);

