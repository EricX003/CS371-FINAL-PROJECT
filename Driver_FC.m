%Prompt for the user
path = input("Please input the path to your data: ", 's');

% Prompt if the user's input is not valid 
while ~isfile(path)
    path = input("Try again, your previous input was not a valid path: ", 's');
end

% Reads in the user's data as a table
data = readtable(path);

% Get's variable name
header = data.Properties.VariableNames;

% Turns table to an array
data = table2array(data);

% Returns a random permutation of the data
data = data(randperm(size(data, 1)), :);

[num_rows, num_cols] = size(data);

label_specified = false;
label_idx = -1;

while ~label_specified
    % Prompt for the user to specify name/index of ground-truth label
    label_index_method = input("Would you like to specify the name or index of the ground-truth label? Please write either name or index: ", 's');
    
    % Switch statement to ensure the validity of the label/index
    switch(label_index_method)
        case "name"
            label_name = input("Please specify the label name (case sensitive): ", 's');
            for idx = 1:num_cols
                if string(header{idx}) == string(label_name)
                    label_idx = idx;
                    label_specified = true;
                end
               
            end
            
            % Prompt if the label name isn't valid
            if ~label_specified
                disp("You have not inputted a valid label name.")
            end
        case "index"
            index = input("Please specify a index (indexing begins at 1): ");
            if index <= num_cols && index >= 1
                label_idx = index;
                label_specified = true;
            end
            
            % Prompt if the index isn't valid
            if ~label_specified
                disp("You have not inputted a valid index.")
            end
        otherwise
            disp("You have not specify either name or index, try again.")

              
    end
end

y = {};
% Gets the minimum and maximum of the data
class_min = min(data(:,label_idx));
class_max = max(data(:,label_idx));
num_classes = class_max - class_min + 1;

for idx = 1:num_rows
    temp_data = zeros([1, num_classes]);
    temp_data(data(idx, label_idx) + (1 - class_min)) = 1;
    y{end + 1} = temp_data;
end

% Puts in data until it reaches the label_idx into an empty array
data(:, label_idx) = [];
x = {};

[length, num_cols] = size(data);

normalize = input("Would you like the data to be normalized (Y or N): ", 's');
while ~(normalize == "Y" || normalize == "N")
    normalize = input("Would you like the data to be normalized (Y or N): ", '');
end

if normalize == "Y"
    for idx = 1:num_cols
        data(:, idx) = (data( : , idx) - mean(data( : , idx)')) ./ std(data( : , idx)');
    end
end

for idx = 1:length
        x{end + 1} = data(idx, :);
    end

[length, num_cols] = size(data);

MSE = @(act, pred) (mean((act - pred) .^ 2) / max(size(act)));
d_MSE = @(act, pred) (2 * (pred - act) /  max(size(act)));

% Appending layers
TestNet = Net(MSE, d_MSE);

% Specifying the activation function's type
act_function_type = input("Please specify an activation function from sigmoid, tanh, relu, or leaky_relu: ", 's');
% Specify neuron ammount 
neurons = input("Please specify the number of neurons in your first hidden layer: ");

TestNet = TestNet.add(FC_Layer(num_cols, neurons));
TestNet = TestNet.add(Activation_Layer(act_function_type));

last = neurons;

% Prompt for the user to add layers
neurons = input("Please specify the number of neurons in your next hidden layer (input -1 at any time to stop adding layers): ");

% Adds the fully connected layers and activation layers with the specified function
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

% Prompt to enter in epoch amount
EPOCHS = input("Please input your desired number of training epochs: ");
while EPOCHS < 1
    EPOCHS = input("Please input your desired number of training epochs: ");
end

% Prompt to enter in the initial learning rate
LR = input("Please input your desired initial learning rate: ");

% Uses the user's desired epochs and LR based on the chosen scheduler below
SCHEDULER = Step_LR_Scheduler(EPOCHS + 1, 1, LR);

% Prompt to choose the type of learning rate scheduler 
scheduler_type = input("Would you like a cyclic scheduler, a step scheduler, or no scheduler (input cyclic, step, or none): ", 's');
switch scheduler_type
    
    % Customizing your chosen learning rate scheduler below
    case "cyclic"
        STEP_SIZE = input("Please input your desired step size: ");
        GAMMA = input("Please input your desired gamma value: ");
        MAX_LR = input("Please input your desired maximum LR: ");
        SCHEDULER = Cyclic_LR_Scheduler(STEP_SIZE, GAMMA, LR, MAX_LR);

    case "step"
        STEP_SIZE = input("Please input your desired step size: ");
        GAMMA = input("Please input your desired gamma value: ");
        SCHEDULER = Step_LR_Scheduler(STEP_SIZE, GAMMA, LR);

end

% Training Model
TestNet = train(TestNet, x(1:split), y(1:split), x(split + 1:end), y(split + 1:end), EPOCHS, LR, SCHEDULER);
