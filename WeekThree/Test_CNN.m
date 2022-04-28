% Test file to ensure our CNN runs

x = {zeros(6), zeros(6) + 1};
y = {[1, 0]; [0, 1]};

test_net = CNN(2);

% Calls upong the functions to use a convolutional layer, max pooling, softmax
test_net = test_net.add(Convolutional_Layer(3, 3));
test_net = test_net.add(MaxPooling(2));
test_net = test_net.add(SoftMax(2 * 2 * 3, 2));

% Tests with 100000 epochs and.001 learning rate
test_net = test_net.train(x, y, 100000, 0.001);
