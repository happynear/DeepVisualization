function obj = getColorPrior(image,k)
    x = reshape(permute(image,[3 1 2]),[3 size(image,1) * size(image,2)]);
    vgg_mean =  [103.939, 116.779, 123.68];
    x = double(x([3 2 1],:));
    x = bsxfun(@minus,x,vgg_mean');
    options = statset('MaxIter',1000);
    while k>0
        try 
            fprintf('color GMM, try clusters=%d...\n',k);
            obj = gmdistribution.fit(x',k,'Options',options);
            break;
        catch
            k = k-1;
        end;
    end;
end