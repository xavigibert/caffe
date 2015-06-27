function [] = view_accuracy()

%Baseline
fname_base = './logs/caffe.ramawks14.umiacs.umd.edu.gibert.log.INFO.20150617-095258.568';
fname1 = './logs/combined1';
fname2 = '/tmp/caffe.ramawks14.umiacs.umd.edu.gibert.log.INFO.20150617-110901.5667';
fname3 = '/tmp/caffe.ramawks14.umiacs.umd.edu.gibert.log.INFO.20150617-143930.30140';

[rb, accb] = load_accuracy(fname_base);
[r1, acc1] = load_accuracy(fname1);
[r2, acc2] = load_accuracy(fname2);
[r3, acc3] = load_accuracy(fname3);

accb = 100*(1-accb);
acc1 = 100*(1-acc1);
acc2 = 100*(1-acc2);
acc3 = 100*(1-acc3);

figure, plot(rb, accb, r1, acc1, r2, acc2, r3, acc3, 'Linewidth', 2);
grid on, title('Test error');
legend('Baseline', 'combined1','20150617-110901','20150617-143930')

%figure, plot(br, medfilt1(b.train_dict_loss, 11), br, b.test_dict_loss, ...
%    r, medfilt1(train_dict_loss, 11), r, test_dict_loss, 'Linewidth', 2);
%grid on, title('Dictionary loss');
%legend('Baseline training', 'Baseline validation', 'Training set', 'Validation set')

%figure, plot(br, medfilt1(b.train_loss, 11), br, b.test_loss, ...
%    r, medfilt1(train_loss, 11), r, test_loss, 'Linewidth', 2);
%grid on, title('Classification loss');
%legend('Baseline training', 'Baseline validation', 'Training set', 'Validation set')

end

function [val] = extract_value(s, pattern)
    k = strfind(s,pattern);
    if isempty(k)
        val = [];
    else
        s = s(k(1)+length(pattern):end);
        val = sscanf(s, '%f');
    end
end

function [test_iter, test_acc] = load_accuracy(fname)
    test_iter = [];
    test_acc = [];
    %test_dict_loss = [];
    %test_loss = [];
    %train_dict_loss = [];
    %train_loss = [];
    curr_test_iter = 0;

    pattern_iteration = 'Iteration ';
    pattern_testing = 'Testing net';
    pattern_test_accuracy = 'Test net output #0: accuracy = ';
    %pattern_test_dict_loss = 'Test net output #1: dict1_loss = ';
    %pattern_test_loss = 'Test net output #2: loss = ';
    %pattern_train_dict_loss = 'Train net output #0: dict1_loss = ';
    %pattern_train_loss = 'Train net output #1: loss = ';

    fd = fopen(fname,'r');
    while ~feof(fd)
        s = fgets(fd);
        val = extract_value(s, pattern_test_accuracy);
        if ~isempty(val)
            test_acc(end+1) = val;
            test_iter(end+1) = curr_test_iter;
        end
        if ~isempty(strfind(s, pattern_testing))
            curr_test_iter = extract_value(s, pattern_iteration);
        end
        %val = extract_value(s, pattern_test_dict_loss);
        %if ~isempty(val), test_dict_loss(end+1) = val; end
        %val = extract_value(s, pattern_test_loss);
        %if ~isempty(val), test_loss(end+1) = val; end
        %val = extract_value(s, pattern_train_loss);
        %if ~isempty(val), train_loss(end+1) = val; end
        %val = extract_value(s, pattern_train_dict_loss);
        %if ~isempty(val), train_dict_loss(end+1) = val; end
    end
    fclose(fd);

    %n = length(test_acc);
    %r = (0:n-1)*500;
end
