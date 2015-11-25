caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
original_prototxt = 'D:\deepLearning\caffe-windows\models\bvlc_googlenet\googlenet.prototxt';
net_weights = 'D:\deepLearning\caffe-windows\models\bvlc_googlenet\thinned.caffemodel';
layer_name = 'inception_4c/output';
pattern_index=400;
channels = 512;
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

vert_num = 32;
hori_num = 16;
total_map = reshape(1:vert_num*hori_num,hori_num,vert_num)';
vert_split = vert_num/16;
hori_split = hori_num/8;
for vert_ind = 13:16
for hori_ind = 1:1
%     disp([layer_name '_' num2str(vert_ind) num2str(hori_ind) '.png']);
%     if exist([layer_name '_' num2str(vert_ind) num2str(hori_ind) '.png'],'file')
%         continue;
%     end;
% vert_ind = 2;
% hori_ind = 2;
caffe.reset_all();
current_map = total_map((vert_ind-1)*vert_split+1:vert_ind*vert_split,(hori_ind-1)*hori_split+1:hori_ind*hori_split)';

height = vert_split * (field_size(1)+border(1)) + border(1);
width = hori_split * (field_size(2)+border(1)) + border(1);

vgg_mean =  [103.939, 116.779, 123.68];
mean_image = permute(repmat(vgg_mean',[1,width,height]),[2,3,1]);
input_data = randn(width, height, 3, 1, 'single')*50;

for need_negative=1:1
original_net_model = fileread(original_prototxt);
original_net_model = strrep(original_net_model,'negative_slope:0#4c','negative_slope:1#4c');

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

if need_negative == 0
    input_data = randn(width, height, 3, 1, 'single')*50;
end;
all_grad = zeros(width, height, 3, 1, 'single');

visualize_net = caffe.Net(visualize_prototxt,net_weights,'test');
visualize_net.blobs(visualize_net.inputs{1}).set_data(input_data);
visualize_net.forward_to(layer_name);
target_blob = visualize_net.blob_vec(visualize_net.name2blob_index(layer_name));
output_data = target_blob.get_data();
backward_mask = zeros(size(output_data,1), size(output_data,2),'uint16');
backward_label = zeros(size(output_data,1), size(output_data,2),'uint16');
backward_data = zeros(size(output_data,1), size(output_data,2),channels, 'uint16');
label_mat = repmat(1:channels,size(output_data,1) * size(output_data,2),1)';
if map_stride(1) > 0
    backward_mask(ceil(map_stride(1)/2+map_border(1)):map_stride(1):end, ceil(map_stride(2)/2+map_border(2)):map_stride(2):end) = 1;
    backward_label(backward_mask==1) = current_map(:);
    for i=1:channels
        backward_data(:,:,i) = (backward_label == i);
    end;
else
    backward_mask(floor(size(backward_mask,1)/2),floor(size(backward_mask,2)/2)) = 1;
end;
item_num = sum(backward_mask(:));
target_blob = visualize_net.blob_vec(visualize_net.name2blob_index(layer_name));
data_blob = visualize_net.blob_vec(visualize_net.name2blob_index('data'));
for i=1:channels
    if i>384 && i<=448
    backward_data(:,:,i) =  backward_data(:,:,i) * 2;
    end;
end;

weight_decay = 0;
use_color_prior = false;
num_cluster=6;
color_prior = 0.5;
long_size = 512;
if need_negative==0
    tv_norm = 0;
    use_image_blur = false;
    use_image_deblur = false;
else
    tv_norm = 0;
    use_image_blur = true;
    use_image_deblur = true;
end;

if need_negative==0
    lr = 10;
else
    lr=200;
end;
max_lr = 50;
momentum = 0.8;
momentum2 = 0.99;
lastgrad = zeros(size(mean_image));
lastgrad2 = zeros(size(mean_image));
last_cost = -9999999999999;
blurred = false;
numLast = 0;

for iter = 1:1000
    bak_data = input_data;
    bak_grad = lastgrad;
    visualize_net.blobs(visualize_net.inputs{1}).set_data(input_data);
    visualize_net.forward_to(layer_name);
    output_data = target_blob.get_data();
    output_data(backward_data==0) = -999;
    output_map = max(output_data,[],3);
    min_cost = min(output_map(backward_mask==1));
    if min_cost>300 && need_negative==0
        disp(min_cost);
        break;
    end;
    cost = sum(output_map(backward_mask==1)) / item_num;
    if need_negative == 0
        for i=1:channels
            if output_map(backward_label == i) > 300
                backward_data(:,:,i) = 0;
                backward_mask(backward_label == i) = 0;
            end;
        end;
        fprintf('iter=%d,lr=%f,this_cost=%f,last_cost=%f,min_cost=%f,min_num=%d\n',iter,lr,cost, last_cost,min_cost,sum(sum(backward_mask==1)));
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
        grad = grad - weight_decay * I;
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
    else
        %%%%%%%%%%%%%%%%%%%%%%%%adam
        lastgrad = (1 - momentum) * grad   + momentum * lastgrad;%/ norm(res(:))
        lastgrad2 = (1 - momentum2) * grad.^2 + momentum2 * lastgrad2;%/ norm(res(:))
        lg_correct = lastgrad ./ (1 - momentum^iter);
        lg2_correct = lastgrad2 ./ (1 - momentum2^iter);
        input_data(:,:,:,1) = input_data(:,:,:,1) + lr * lg_correct ./ (sqrt(lg2_correct) + 1e-8);
    end;
%     input_data(:,:,:,1) = input_data(:,:,:,1) / norm(input_data(:));
    
    k = mod(iter,10);
    if k==1
        H = fspecial('gaussian',[7 7],1.2);
        if use_image_blur
            input_data(:,:,:,1) = imfilter(input_data(:,:,:,1),H,'same');
            blurred = true;
        end
    end;
    if k==6
        H = fspecial('gaussian',[7 7],1);
        if use_image_deblur
            input_data(:,:,:,1) = deconvlucy(input_data(:,:,:,1), H);
            blurred = false;
        end;
    end;
    
    if (need_negative==1 && cost>last_cost)
        last_cost = cost;
        numLast = 0;
    else
        numLast = numLast + 1;
    end;
    if (need_negative==0 && min_cost>last_cost)
        last_cost = min_cost;
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
    
    %%%%%%%%%%%%%%%%%%%%%%gd
    
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
end;
% imwrite(uint8(output),[layer_name '_' num2str(vert_ind) num2str(hori_ind) '.png']);
end;
end;