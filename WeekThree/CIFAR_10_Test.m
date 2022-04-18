% This unpacked the tar.gz file 
% data = "cifar-10-matlab.tar.gz";
% cifar_10Images = untar(data)';

images = {};
data = reshape(data, [10000, 32*32*3]);
for idx = 1:10000
    images{end + 1} = (double(data(idx, :)) / 256) - 0.5;
end
ground_truth = {};
for label = labels
    temp = zeros([1, 10]);
    temp(label + 1) = 1;
    ground_truth{end + 1} = temp;
end

% Mean Square Error, Loss function, and its derivative
MSE = @(act, pred) (mean((act - pred) .^ 2));
d_MSE = @(act, pred) (2 * (pred - act) /  length(act));

% Appending layers
TestNet = Net(MSE, d_MSE);
TestNet = TestNet.add(FC_Layer(32*32*3, 1024));
TestNet = TestNet.add(Activation_Layer('sigmoid'));
TestNet = TestNet.add(FC_Layer(1024, 10));
TestNet = TestNet.add(Activation_Layer('sigmoid'));

% Training Model
TestNet = train(TestNet, images, ground_truth, 10000, 0.01);

% Outputs the prediction
out = TestNet.predict(x_train)


% cifar_10Images = {load('data_batch_1.mat'), load('data_batch_2.mat'), ... 
%    load('data_batch_3.mat'), load('data_batch_4.mat'), ...
%    load('data_batch_5.mat')};
% 
% testBatch = {load('test_batch.mat')}
% 
% celldisp(testBatch)
