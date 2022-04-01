% This file contains the XOR Test, which validates our framework's soundness and performance

% Data
x_train = {[0, 0]; [0, 1]; [1, 0]; [1, 1]};
y_train = {[0, 1]; [1, 0]; [1, 0]; [0, 1]};

% Mean Square Error, Loss function, and its derivative
MSE = @(act, pred) (mean((act - pred) .^ 2));
d_MSE = @(act, pred) (2 * (pred - act) /  length(act));

% Appending layers
TestNet = Net(MSE, d_MSE);
TestNet = TestNet.add(FC_Layer(2, 5));
TestNet = TestNet.add(Activation_Layer('relu'));
TestNet = TestNet.add(FC_Layer(5, 2));
TestNet = TestNet.add(Activation_Layer('relu'));

% Training Model
TestNet = train(TestNet, x_train, y_train, 10000, 0.01);

% Outputs the prediction
out = TestNet.predict(x_train)
