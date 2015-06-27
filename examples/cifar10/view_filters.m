f = fopen('conv1_blob0.bin');
s=5;
n1=8;
n2=4;
im = ones(2+(s+2)*n2, 2+(s+2)*n1, 3);
for i = 0:n2-1
    for j = 0:n1-1
        im2 = fread(f, [s s], 'float')';
        im2(:,:,2) = fread(f, [s s], 'float')';
        im2(:,:,3) = fread(f, [s s], 'float')';
        im2 = (im2 - mean(im2(:))) / std(im2(:)) + 0.5;
        im2 = im2/3;
        im(3+i*(s+2):(s+2)+i*(s+2),3+j*(s+2):(s+2)+j*(s+2),1:3) = im2;
    end
end
fclose(f);
im3 = imresize(im, 2, 'nearest');
imwrite(im3, 'learned_filters.png');