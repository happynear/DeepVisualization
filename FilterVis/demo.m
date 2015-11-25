original_prototxt = 'googlenet.prototxt';
net_weights = 'thinned.caffemodel';
layer_name = 'inception_4c/output';
channels = 512;
filter_id = 1;

param.weight_decay = 0;
param.tv_norm = 0;
param.use_image_blur = true;
param.use_image_deblur = false;
param.learning_rate = 400;

ShowMidFilter(original_prototxt, net_weights, layer_name, channels, filter_id, param);