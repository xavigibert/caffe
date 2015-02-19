function [iters, loss_train, loss_test, acc_test] = view_loss(fname)

% [iters, loss_train, loss_test, acc_test] = view_loss('/tmp/caffe.ramawks14.umiacs.umd.edu.gibert.log.INFO.20150113-144923.14818');
% [diters, dloss_train, dloss_test, dacc_test] = view_loss('/tmp/caffe.ramawks14.umiacs.umd.edu.gibert.log.INFO.20150113-153511.13599');

iters = [];
loss_train = [];
loss_test = [];
acc_test = [];

f = fopen(fname);
tline = fgetl(f);
iteration = 0;
tmp_loss_train = [];
while ischar(tline)
    idx = strfind(tline, 'Iteration');
    if ~isempty(idx)
        iteration = sscanf(strtok(tline(idx+10:end), ','),'%d');
    end
    istrain = ~isempty(strfind(tline, 'Train net'));
    istest = ~isempty(strfind(tline, 'Test net'));
    idx = strfind(tline, 'loss = ');
    if istrain && ~isempty(idx)
        tmp_loss_train(end+1) = sscanf(strtok(tline(idx+7:end), ' '),'%f');
    elseif istest && ~isempty(idx)
        iters(end+1) = iteration;
        loss_train(end+1) = mean(tmp_loss_train);
        tmp_loss_train = [];
        loss_test(end+1) = sscanf(strtok(tline(idx+7:end), ' '),'%f');
    end
    idx = strfind(tline, 'accuracy = ');
    if istest && ~isempty(idx) && ~isempty(iters);
        acc_test(end+1) = sscanf(tline(idx+11:end),'%f');
    end
    
    tline = fgetl(f);
end

figure(1)
plot(iters, loss_train, iters, loss_test);
figure(2)
plot(iters, acc_test);