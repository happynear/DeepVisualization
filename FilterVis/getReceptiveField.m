function [field_size, field_stride] = getReceptiveField(original_prototxt, net_weights, layer_name, pattern_index, init_size, relu_flag)
%GetRecpetiveField Get the receptive filed size of neurons in a specified layer
%   此处显示详细说明
%     original_prototxt = 'VGG_16_nueralart.prototxt';
%     net_weights = 'D:\deepLearning\caffe-windows\matlab\demo\NeuralArt\VGG16_thinned_net.caffemodel';
%     layer_name = 'conv2_2';
%     pattern_index = 1;
    if nargin==4
        init_size = [512 512];
        relu_flag = '';
    end;
    if nargin==5
        relu_flag = '';
    end;
    addpath('./PrototxtGen');
    original_net_model = fileread(original_prototxt);
%     original_net_model = strrep(original_net_model,'MAX','AVE');
%     original_net_model = strrep(original_net_model,'type: "ReLU"',sprintf('type: "ReLU"\r\n  relu_param{\r\n    negative_slope:1\r\n  }'));
    original_net_model = strrep(original_net_model,['negative_slope:0#' relu_flag],['negative_slope:1#' relu_flag]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%get style pattern
    receptive_prototxt = strrep(original_prototxt,'.prototxt','_receptive.prototxt');
    height = init_size(1);
    width = init_size(2);

    fid = fopen(receptive_prototxt,'w');
    proto_txt{1} = 'name: "Receptive"';
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
    
    receptive_net = caffe.Net(receptive_prototxt,net_weights,'test');
    
    im_data = randn(height, width, 3, 1, 'single')*50;
    
    receptive_net.blobs(receptive_net.inputs{1}).set_data(im_data);
    receptive_net.forward_to(layer_name);
    target_blob = receptive_net.blob_vec(receptive_net.name2blob_index(layer_name));
    output_data = target_blob.get_data();
    
    output_data(:) = 0;
    output_data(floor(size(output_data,1)/2), floor(size(output_data,2)/2), pattern_index) = 1;
    
    target_blob.set_diff(output_data);
    receptive_net.backward_from(layer_name);
    data_blob = receptive_net.blob_vec(receptive_net.name2blob_index('data'));
    data_diff = data_blob.get_diff();
    data_diff = sum(abs(data_diff),3);
%     data_diff(abs(data_diff)<abs(data_diff(floor(height/2),floor(width/2))) / 10) = 0;
    
    field_size = zeros(1,2);
    field_stride = zeros(1,2);
    
    height_index1 = find(sum(data_diff,2)>0);
    field_size(1) = max(height_index1) - min(height_index1) + 1;
    width_index1 = find(sum(data_diff)>0);
    field_size(2) = max(width_index1) - min(width_index1) + 1;
    
    output_data(:) = 0;
    output_data(floor(size(output_data,1)/2)+1, floor(size(output_data,2)/2)+1, pattern_index) = 1;
    target_blob.set_diff(output_data);
    receptive_net.backward_from(layer_name);
    data_blob = receptive_net.blob_vec(receptive_net.name2blob_index('data'));
    data_diff = data_blob.get_diff();
    data_diff = sum(abs(data_diff),3);
%     data_diff(abs(data_diff)<abs(data_diff(floor(height/2),floor(width/2))) / 10) = 0;
     
    height_index2 = find(sum(data_diff,2)>0);
    width_index2 = find(sum(data_diff)>0);
    field_stride(1) = min(height_index2) - min(height_index1);
    field_stride(2) = min(width_index2) - min(width_index1);