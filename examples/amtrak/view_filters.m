f = fopen('/scratch0/gitlab/railwayvision/libraries/caffe/blob0.bin');
im = ones(2+11*4, 2+11*8);
for i = 0:3
    for j = 0:7
        im2 = fread(f, [9 9], 'float')';
        im2 = (im2 - mean(im2(:))) / std(im2(:)) + 0.5;
        im2 = im2/3;
        im(3+i*11:11+i*11,3+j*11:11+j*11) = im2;
    end
end
fclose(f);
im3 = imresize(im, 2, 'nearest');
imwrite(im3, 'learned_filters.png');