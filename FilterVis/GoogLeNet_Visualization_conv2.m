caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
% original_prototxt = 'googlenet_neuralart_conv2.prototxt';
% net_weights = 'D:\deeplearning\caffe-windows\examples\GoogLeNet\imagenet_googlenet.caffemodel';
% layer_name = 'conv2';
original_prototxt = 'D:\deepLearning\caffe-windows\models\bvlc_googlenet\googlenet.prototxt';
net_weights = 'D:\deepLearning\caffe-windows\models\bvlc_googlenet\thinned.caffemodel';
layer_name = 'conv2/3x3';
channels = 192;
pattern_index = 1;
initial_size = [256 256];
[field_size, field_stride] = getReceptiveField(original_prototxt, net_weights, layer_name, pattern_index, initial_size);
if max(field_size) > initial_size(1)*2/3
%     error('field_size:(%d,%d),field_stride:(%d,%d), please increase the initial size.',field_size(1),field_size(2),field_stride(1),field_stride(2));
    map_stride = zeros(1,2);
else
    map_stride = ceil((field_size + 7) ./ field_stride);
end;

height = 230;
width = 230 * 3;

for need_negative=1:1
original_net_model = fileread(original_prototxt);

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

vgg_mean =  [103.939, 116.779, 123.68];
mean_image = permute(repmat(vgg_mean',[1,width,height]),[2,3,1]);
    input_data = randn(width, height, 3, 1, 'single');
all_grad = zeros(width, height, 3, 1, 'single');

visualize_net = caffe.Net(visualize_prototxt,net_weights,'test');
visualize_net.blobs(visualize_net.inputs{1}).set_data(input_data);
visualize_net.forward_to(layer_name);
target_blob = visualize_net.blob_vec(visualize_net.name2blob_index(layer_name));
output_data = target_blob.get_data();
backward_mask = zeros(size(output_data,1), size(output_data,2),'uint8');
backward_label = zeros(size(output_data,1), size(output_data,2),'uint8');
backward_data = zeros(size(output_data,1), size(output_data,2),channels, 'uint8');
label_mat = repmat(1:channels,size(output_data,1) * size(output_data,2),1)';
if map_stride(1) > 0
    backward_mask(floor(map_stride(1)/2+field_stride(1)/2):map_stride(1):end, floor(map_stride(2)/2+field_stride(2)/2):map_stride(2):end) = 1;
    backward_label(backward_mask==1) = label_mat(1:sum(sum(backward_mask==1)));
    for i=1:channels
        backward_data(:,:,i) = (backward_label == i);
    end;
else
    backward_mask(floor(size(backward_mask,1)/2),floor(size(backward_mask,2)/2)) = 1;
end;
item_num = sum(backward_mask(:));
target_blob = visualize_net.blob_vec(visualize_net.name2blob_index(layer_name));
data_blob = visualize_net.blob_vec(visualize_net.name2blob_index('data'));

target_layer = visualize_net.layers(layer_name);
target_weight = target_layer.params(1).get_data();
target_weight_hist = sum(sum(sum(abs(target_weight),1),2),3);
figure(1);
bar(target_weight_hist(:));

weight_decay = 0;
use_color_prior = false;
num_cluster=6;
color_prior = 0.5;
long_size = 512;
    tv_norm = 0;
    use_image_blur = true;
    use_image_deblur = true;
    weight_decay = 0;
    lr=100;
    max_lr = 200;

momentum = 0.8;
momentum2 = 0.99;
lastgrad = zeros(size(mean_image));
lastgrad2 = zeros(size(mean_image));
last_cost = -9999999999999;
if need_negative==0
    maxiter = 1000;
else
    maxiter = 100;
end;

for iter = 1:1000
    bak_data = input_data;
    bak_grad = lastgrad;
    visualize_net.blobs(visualize_net.inputs{1}).set_data(input_data);
    visualize_net.forward_to(layer_name);
    output_data = target_blob.get_data();
    output_data(backward_data==0) = -999;
    output_map = max(output_data,[],3);
    min_cost = min(output_map(backward_mask==1));
    if min_cost>100 && need_negative==0
        disp(min_cost);
        break;
    end;
    cost = sum(output_map(backward_mask==1)) / item_num;
    if need_negative == 0
        for i=1:channels
            if output_map(backward_label == i) > 100
                backward_data(:,:,i) = 0;
                backward_mask(backward_label == i) = 0;
            end;
        end;
        fprintf('iter=%d,lr=%f,this_cost=%f,last_cost=%f,min_cost=%f,min_num=%d\n',iter,lr,cost, last_cost,min_cost,sum(backward_mask(:)));
    else
        fprintf('iter=%d,lr=%f,this_cost=%f,last_cost=%f,image_norm=%f,min_cost=%f\n',iter,lr,cost, last_cost,norm(input_data(:)),min_cost);
    end;
    
%     output_data(:) = 0;
%     output_data(:,:,pattern_index) = backward_mask;
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
        grad = grad - weight_decay * input_data(:,:,:,1);
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
    if need_negative==1
        %%%%%%%%%%%%%%%%%%%%%%%%gd linear search
        lastgrad = (1 - momentum) * lr * grad   + momentum * lastgrad;%/ norm(res(:))
        input_data(:,:,:,1) = input_data(:,:,:,1) + lastgrad;
        if cost>last_cost
        last_cost = cost;
    end;
    else
        %%%%%%%%%%%%%%%%%%%%%%%%adam
        lastgrad = (1 - momentum) * grad   + momentum * lastgrad;%/ norm(res(:))
        lastgrad2 = (1 - momentum2) * grad.^2 + momentum2 * lastgrad2;%/ norm(res(:))
        lg_correct = lastgrad ./ (1 - momentum^iter);
        lg2_correct = lastgrad2 ./ (1 - momentum2^iter);
        input_data(:,:,:,1) = input_data(:,:,:,1) + lr * lg_correct ./ (sqrt(lg2_correct) + 1e-8);
%         lastgrad = (1 - momentum) * lr * grad   + momentum * lastgrad;%/ norm(res(:))
%         input_data(:,:,:,1) = input_data(:,:,:,1) + lastgrad;
        if min_cost<last_cost
            lr = lr * 0.9;
        end;
        if min_cost>last_cost + 1e-5&&lr<max_lr
            lr = lr * 1.1;
        end;
        if min_cost>last_cost
            last_cost = min_cost;
        end;
    end;
%     input_data(:,:,:,1) = input_data(:,:,:,1) / norm(input_data(:));
    
    k = mod(iter,10);
    if k==0
        H = fspecial('gaussian',[5 5],0.6);
        if use_image_blur
            input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
        end
    end;
    if k==5
        H = fspecial('gaussian',[5 5],0.5);
        if use_image_deblur
            input_data(:,:,:,1) = deconvlucy(input_data(:,:,:,1), H);
        end;
    end;
    
%     if cost<last_cost
%         lr = lr * 0.9;
% %         input_data = bak_data;%why...
% %         last_grad = bak_grad;
%     end;
%     if cost>last_cost&&lr<max_lr%&& (this_prob-last_prob) / this_prob < 0.001
%         lr = lr * 1.1;
%     end;
    
    if lr<1e-5
        break;
    end;
    
    %%%%%%%%%%%%%%%%%%%%%%gd
    
    if mod(iter,10)==0
        output = mean_image + input_data(:,:,:,1);
        output = output(:, :, [3, 2, 1]);
        output = permute(output, [2 1 3]);
        figure(3);
        imshow(uint8(output));
        title('generated image');
        I = output;
    end;
end;
end;