path = input("Please input the path to your data, enclosed in double quotes: ");

while ~isfile(path)
    path = input("Try again, your previous input was not a valid path: ", 's');
end

data = readtable(path);

header = data.Properties.VariableNames;

data = table2array(data);

data = data(randperm(size(data, 1)), :);

[num_rows, num_cols] = size(data);

label_specified = false;
label_idx = -1;

while ~label_specified
    label_index_method = input("Would you like to specify the name or index of the ground-truth label? Please write either name or index: ", 's');

    switch(label_index_method)
        case "name"
            label_name = input("Please specify the label name (case sensitive): ", 's');
            for idx = 1:num_cols
                if header{idx} == string(label_name)
                    label_idx = idx;
                    label_specified = true;
                end
               
            end
            if ~label_specified
                disp("You have not inputted a valid label name.")
            end
        case "index"
            index = input("Please specify a index (indexing begins at 1): ");
            if index <= num_cols && index >= 1
                label_idx = index;
                label_specified = true;
            end
            if ~label_specified
                disp("You have not inputted a valid index.")
            end
        otherwise
            disp("You have not specify either name or index, try again.")

              
    end
end

y = {};
class_min = min(data(:,label_idx));
class_max = max(data(:,label_idx));
num_classes = class_max - class_min + 1;

for idx = 1:num_rows
    temp_data = zeros([1, num_classes]);
    temp_data(data(idx, label_idx) + (1 - class_min)) = 1;
    y{end + 1} = temp_data;
end

data(:, label_idx) = [];
x = {};

[length, num_cols] = size(data);

for idx = 1:num_cols
    data(:, idx) = (data( : , idx) - mean(data( : , idx)')) ./ std(data( : , idx)');
end

for idx = 1:length
    x{end + 1} = data(idx, :);
end

[length, num_cols] = size(data);

MSE = @(act, pred) (mean((act - pred) .^ 2) / max(size(act)));
d_MSE = @(act, pred) (2 * (pred - act) /  max(size(act)));

% Appending layers
TestNet = Net(MSE, d_MSE);

act_function_type = input("Please specify an activation function from sigmoid, tanh, relu, or leaky_relu: ", 's');
neurons = input("Please specify the number of neurons in your first hidden layer: ");

TestNet = TestNet.add(FC_Layer(num_cols, neurons));
TestNet = TestNet.add(Activation_Layer(act_function_type));

last = neurons;

neurons = input("Please specify the number of neurons in your next hidden layer (input -1 at any time to stop adding layers): ");

while neurons > 0
    TestNet = TestNet.add(FC_Layer(last, neurons));
    TestNet = TestNet.add(Activation_Layer(act_function_type));
    last = neurons;
    neurons = input("Please specify the number of neurons in your next hidden layer (input -1 at any time to stop adding layers): ");
end

TestNet = TestNet.add(FC_Layer(last, num_classes));
TestNet = TestNet.add(Activation_Layer(act_function_type));

train_prop = input("Please input your target training proportion: ");
while train_prop > 1 || train_prop < 0
    train_prop = input("Please input your target training proportion: ");
end
split = floor(train_prop * length);

EPOCHS = input("Please input your desired number of training epochs: ");
while EPOCHS < 1
    EPOCHS = input("Please input your desired number of training epochs: ");
end

LR = input("Please input your desired initial learning rate: ");

% Training Model
TestNet = train(TestNet, x(1:split), y(1:split), x(split + 1:end), y(split + 1:end), EPOCHS, LR);

% Outputs the prediction
out = TestNet.predict(x)