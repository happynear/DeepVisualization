function [cost, grad] = NeuralArtCost(input_data,input_size,stylegen_net,forward_input,style_weights)
if ~isempty(input_size)
    input_data = reshape(input_data,input_size);
end;
forward_input(1) = {input_data};
prob = stylegen_net.forward(forward_input);
i=1;
while i<=length(prob)
    if length(prob{i})>1
        prob(i)=[];
    end;
    i=i+1;
end;
cost = [prob{:}] * style_weights' / 1000;
stylegen_net.backward_prefilled();
grad = stylegen_net.blob_vec(stylegen_net.name2blob_index('data')).get_diff();
if ~isempty(input_size)
    grad = grad(:);
end;