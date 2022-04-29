% Test file to ensure our CNN runs

x = {zeros(6) + 1, zeros(6) + 1};
y = {[1, 0]; [0, 1]};

test_net = CNN(2);

% Calls upon the functions to ensure they work
test_net = test_net.add(Convolutional_Layer(3, 10));
test_net = test_net.add(MaxPooling(2));
test_net = test_net.add(SoftMax(2 * 2 * 10, 2));

% Trains the network with the given epochs and initial learning rate 
test_net = test_net.train(x, y, 100000, 0.001);
