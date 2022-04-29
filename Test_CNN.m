x = {zeros(6) + 1, zeros(6) + 1};
y = {[1, 0]; [0, 1]};

test_net = CNN(2);

test_net = test_net.add(Convolutional_Layer(3, 10));
test_net = test_net.add(MaxPooling(2));
test_net = test_net.add(SoftMax(2 * 2 * 10, 2));

test_net = test_net.train(x, y, 100000, 0.001);