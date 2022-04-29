% This file contains the XOR Test, which validates our framework's soundness and performance

% Data
x_train = {[0, 0]; [0, 1]; [1, 0]; [1, 1]};
y_train = {[0, 1]; [1, 0]; [1, 0]; [0, 1]};

% Mean Square Error, Loss function, and its derivative
MSE = @(act, pred) (mean((act - pred) .^ 2) / max(size(act)));
d_MSE = @(act, pred) (2 * (pred - act) /  max(size(act)));

% Appending layers
TestNet = Net(MSE, d_MSE);
TestNet = TestNet.add(FC_Layer(2, 5));
TestNet = TestNet.add(Activation_Layer('leaky_relu'));
TestNet = TestNet.add(FC_Layer(5, 2));
TestNet = TestNet.add(Activation_Layer('leaky_relu'));

SCHEDULER = Step_LR_Scheduler(5, 0.5, 0.01);

% Training Model
TestNet = train(TestNet, x_train, y_train, x_train, y_train, 10000, 0.01, SCHEDULER);

disp("Predicted: ")
disp(TestNet.predict(x_train))
disp("Actual: ")
disp(y_train)