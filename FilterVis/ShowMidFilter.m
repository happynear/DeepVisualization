function ShowMidFilter(original_prototxt, net_weights, layer_name, channels, filter_id, param)
% Visualize the mid-level filters. Based on paper:
% Feng Wang, Haijun Liu, Jian Cheng, 
% Visualizing Deep Neural Network by Alternately Image Blurring and Deblurring
%
% original_prototxt: The path to the network definition file. Please refer
%   to the googlenet.prototxt to modify your network.
% net_weights: The trained network model.
% layer_name: The layer name which we are willing to visualize.
% channels: how many feature maps in this layer. Maybe we can remove this
%   parameter in the future.
% filter_id: Which filter we want to visualize. It should not be greater
%   than channels.
% param: Some hyper-paramters to optimize the network. See demo.m for
%   detail list.
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
% original_prototxt = 'D:\deepLearning\caffe-windows\models\bvlc_googlenet\googlenet.prototxt';
% net_weights = 'D:\deepLearning\caffe-windows\models\bvlc_googlenet\thinned.caffemodel';
% layer_name = 'inception_4c/output';
pattern_index=1;
% channels = 512;
initial_size = [400 400];
[field_size, field_stride] = getReceptiveField(original_prototxt, net_weights, layer_name, pattern_index, initial_size);
field_size = ceil(field_size ./ field_stride) .* field_stride;
border = field_stride(1);
map_border = border ./ field_stride;
if max(field_size) > 224
%     error('field_size:(%d,%d),field_stride:(%d,%d), please increase the initial size.',field_size(1),field_size(2),field_stride(1),field_stride(2));
    field_size = [224 224];
    field_size = ceil(field_size ./ field_stride) .* field_stride;
    map_stride = ceil((field_size + border) ./ field_stride);
else
    map_stride = ceil((field_size + border) ./ field_stride);
end;

caffe.reset_all()
vert_num = 1;
hori_num = 1;
height = vert_num * (field_size(1)+border(1)) + border(1);
width = hori_num * (field_size(2)+border(1)) + border(1);

vgg_mean =  [103.939, 116.779, 123.68];
mean_image = permute(repmat(vgg_mean',[1,width,height]),[2,3,1]);
input_data = randn(width, height, 3, 1, 'single');

original_net_model = fileread(original_prototxt);
original_net_model = strrep(original_net_model,['negative_slope:0#' layer_name(11:12)],['negative_slope:1#' layer_name(11:12)]);

visualize_prototxt = strrep(original_prototxt,'.prototxt','_visualize.prototxt');

fid = fopen(visualize_prototxt,'w');
proto_txt{1} = 'name: "Visualize"';
proto_txt{2} = 'input: "data"';
proto_txt{3} = 'input_dim: 1';
proto_txt{4} = 'input_dim: 3';
proto_txt{5} = ['input_dim: ' num2str(height)];
proto_txt{6} = ['input_dim: ' num2str(width)];
for i=1:6
    fprintf(fid,'%s\r\n',proto_txt{i});
end;

fprintf(fid,'%s\r\n',original_net_model);
fclose(fid);

all_grad = zeros(width, height, 3, 1, 'single');

visualize_net = caffe.Net(visualize_prototxt,net_weights,'test');
visualize_net.blobs(visualize_net.inputs{1}).set_data(input_data);
visualize_net.forward_to(layer_name);
target_blob = visualize_net.blob_vec(visualize_net.name2blob_index(layer_name));
output_data = target_blob.get_data();
backward_mask = zeros(size(output_data,1), size(output_data,2),'uint16');
backward_label = zeros(size(output_data,1), size(output_data,2),'uint16');
backward_data = zeros(size(output_data,1), size(output_data,2),channels, 'uint16');
if map_stride(1) > 0
    backward_mask(ceil(map_stride(1)/2+map_border(1)):map_stride(1):end, ceil(map_stride(2)/2+map_border(2)):map_stride(2):end) = 1;
    backward_label(backward_mask==1) = filter_id;
    for i=1:channels
        backward_data(:,:,i) = (backward_label == i);
    end;
else
    backward_mask(floor(size(backward_mask,1)/2),floor(size(backward_mask,2)/2)) = 1;
end;
item_num = sum(backward_mask(:));
target_blob = visualize_net.blob_vec(visualize_net.name2blob_index(layer_name));
data_blob = visualize_net.blob_vec(visualize_net.name2blob_index('data'));

if isfield(param,'weight_decay')
    weight_decay = param.weight_decay;
else
    weight_decay = 0;
end;
if isfield(param,'tv_norm')
    tv_norm = param.tv_norm;
else
    tv_norm = 0;
end;
if isfield(param,'use_image_blur')
    use_image_blur = param.use_image_blur;
else
    use_image_blur = true;
end;
if isfield(param,'use_image_deblur')
    use_image_deblur = param.use_image_deblur;
else
    use_image_deblur = true;
end;
if isfield(param,'learning_rate')
    lr = param.learning_rate;
else
    lr=200;
end;

momentum = 0.8;
momentum2 = 0.99;
lastgrad = zeros(size(mean_image));
lastgrad2 = zeros(size(mean_image));
last_cost = -9999999999999;
numLast = 0;

for iter = 1:1000
    visualize_net.blobs(visualize_net.inputs{1}).set_data(input_data);
    visualize_net.forward_to(layer_name);
    output_data = target_blob.get_data();
    output_data(backward_data==0) = -999;
    output_map = max(output_data,[],3);
    min_cost = min(output_map(backward_mask==1));
    cost = sum(output_map(backward_mask==1)) / item_num;
    fprintf('iter=%d,lr=%f,this_cost=%f,last_cost=%f,image_norm=%f,min_cost=%f\n',iter,lr,cost, last_cost,norm(input_data(:)),min_cost);

    target_blob.set_diff(backward_data);
    visualize_net.backward_from(layer_name);
    grad = data_blob.get_diff();
    all_grad = all_grad + grad;
    if iter==20
        input_data(all_grad==0) = 0;
    end;
    
    if tv_norm > 0
        I = input_data(:,:,:,1);
        Gx = (I(2:end-1,:,:) - I(1:end-2,:,:)) - (I(3:end,:,:) - I(2:end-1,:,:));
        Gx = [(I(1,:,:) - I(2,:,:)); Gx; (I(end,:,:) - I(end-1,:,:))];
        Gy = (I(:,2:end-1,:) - I(:,1:end-2,:)) - (I(:,3:end,:) - I(:,2:end-1,:));
        Gy = [(I(:,1,:) - I(:,2,:)) Gy (I(:,end,:) - I(:,end-1,:))];
        grad = grad - tv_norm * (Gx+Gy);
    end;
    
    if weight_decay > 0
        grad = grad - weight_decay * I;
    end;
    
    if true
        %%%%%%%%%%%%%%%%%%%%%%%%gd linear search
        lastgrad = (1 - momentum) * lr * grad   + momentum * lastgrad;%/ norm(res(:))
        input_data(:,:,:,1) = input_data(:,:,:,1) + lastgrad;
    else
        %%%%%%%%%%%%%%%%%%%%%%%%adam
        lastgrad = (1 - momentum) * grad   + momentum * lastgrad;%/ norm(res(:))
        lastgrad2 = (1 - momentum2) * grad.^2 + momentum2 * lastgrad2;%/ norm(res(:))
        lg_correct = lastgrad ./ (1 - momentum^iter);
        lg2_correct = lastgrad2 ./ (1 - momentum2^iter);
        input_data(:,:,:,1) = input_data(:,:,:,1) + lr * lg_correct ./ (sqrt(lg2_correct) + 1e-8);
    end;
    
    k = mod(iter,10);
    if k==1
        H = fspecial('gaussian',[7 7],0.8);
        if use_image_blur
            input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
            blurred = true;
        end
    end;
    if k==6
        H = fspecial('gaussian',[7 7],0.6);
        if use_image_deblur
            input_data(:,:,:,1) = deconvlucy(input_data(:,:,:,1), H);
            blurred = false;
        end;
    end;
    
    if cost>last_cost
        last_cost = cost;
        numLast = 0;
    else
        numLast = numLast + 1;
    end;
    if numLast>100
        break;
    end;
    if lr<1e-5
        break;
    end;
    
    if mod(iter,100)==0
        output = mean_image + input_data(:,:,:,1);
        output = output(:, :, [3, 2, 1]);
        output = permute(output, [2 1 3]);
        figure(3);
        imshow(uint8(output));
%         title('generated image');
        I = output;
    end;
end;