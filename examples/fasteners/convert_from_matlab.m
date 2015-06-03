load training.mat

max_items = 2200;

% Spilt file names
mkdir('../../data/fasteners');
template_fname_img = '../../data/fasteners/split%d-images-idx3-ubyte';    % Splits number from 0 to 4
template_fname_lbl = '../../data/fasteners/split%d-labels-idx1-ubyte';

% This dataset is too unbalanced. We need to remove class 9 due to
% insufficient number of samples. Class 10 will become class 9
sel = (clIdx~=9);
fv = fv(sel,:);
fvM = fvM(sel,:);
clIdx = clIdx(sel);
fold = fold(sel);
imgIdx = imgIdx(sel);
clIdx(clIdx==10) = 9;
fastIsDefective = fastIsDefective((0:10)~=9);
fastIsSymmetric = fastIsSymmetric((0:10)~=9);

% Train each class
num_classes_wbg = max(clIdx)+2;
num_models = max(fold)+1;

% To balance number of samples per class
min_samples_per_class = ceil(max_items / num_classes_wbg);

sym = [true fastIsSymmetric];

% Generate split
for splitIdx = 0:num_models-1
    fname_img = sprintf(template_fname_img, splitIdx);
    fname_lbl = sprintf(template_fname_lbl, splitIdx);
    fid_img = fopen(fname_img, 'w');
    if fid_img < 0
        error(['ERROR: Failed to open output file ' fname_img]);
    end
    fid_lbl = fopen(fname_lbl, 'w');
    if fid_img < 0
        fclose(fid_img);
        error(['ERROR: Failed to open output file ' fname_lbl]);
    end
    
    % Select data for split performing data augmentation for symmetric classes
    uimg = fv(fold==splitIdx,:);
    uimgM = fv(fold==splitIdx,:);
    ulbl = clIdx(fold==splitIdx,:)+1;
    img = [];
    lbl = [];
    for ci = 0:num_classes_wbg-1
        timg = [];
        tlbl = [];
        if nnz(ulbl==ci) == 0
            error('Insufficient number of samples');
        end
        while length(tlbl) < min_samples_per_class
            timg = [timg; uimg(ulbl==ci,:)];
            tlbl = [tlbl; ulbl(ulbl==ci)];
            if sym(ci+1)
                % Add symmetric samples
                timg = [timg; uimgM(ulbl==ci,:)];
                tlbl = [tlbl; ulbl(ulbl==ci)];
            end
        end
        img = [img; timg(1:min_samples_per_class,:)];
        lbl = [lbl; tlbl(1:min_samples_per_class)];
    end
    
    % Perform random permutation
    perm = randperm(size(img,1));
    img = img(perm,:);
    lbl = lbl(perm);
    
    num_items = uint32(size(img,1));
    num_rows = uint32(80);
    num_cols = uint32(80);
    
    fprintf('Split %d, num_items=%d\n', splitIdx, num_items);
    if num_items < max_items
        error('Insufficient number of items');
    end
    % Truncate data
    num_items = uint32(max_items);
    img = img(1:num_items,:);
    lbl = lbl(1:num_items);
    
    % Save headers
    fwrite(fid_img, swapbytes(uint32(2051)), 'uint32');  % Magic number
    fwrite(fid_img, swapbytes(num_items), 'uint32');
    fwrite(fid_img, swapbytes(num_rows), 'uint32');
    fwrite(fid_img, swapbytes(num_cols), 'uint32');
    
    fwrite(fid_lbl, swapbytes(uint32(2049)), 'uint32');  % Magic number
    fwrite(fid_lbl, swapbytes(num_items), 'uint32');
    
    % Save data
    img = img';     % Transpose so image coordinate is the major unit
    fwrite(fid_img, uint8(img(:)*4+128), 'uint8');
    fwrite(fid_lbl, uint8(lbl), 'uint8');
 
    fclose(fid_img);
    fclose(fid_lbl);
end

