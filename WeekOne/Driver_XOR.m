x_train = [[[0,0]]; [[0,1]]; [[1,0]]; [[1,1]]];
y_train = [[[0]]; [[1]]; [[1]]; [[0]]];

MSE = @(act, pred) mean((act - pred) .^ 2);
d_MSE = @(act, pred) (2 * (pred - act) / length(act));

TestNet = Net(MSE, d_MSE);
TestNet.add(FC_Layer(2, 5));
TestNet.add(ReLU());
TestNet.add(FC_Layer(5, 1));
TestNet.add(ReLU());

TestNet.train(x_train, y_train, 1000, 0.1);

out = TestNet.predict(x_train);
