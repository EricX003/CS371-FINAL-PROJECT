% Maintains cell array of layers, has capabilities to initialize, append layers, train, and predict
classdef Net
    
    % List of needed variables
    properties
        layers
        loss
        d_loss
    end

    methods
        
        % Contains all the layers 
        function obj = Net(loss, d_loss)
            obj.layers = {};
            obj.loss = loss;
            obj.d_loss = d_loss;
        end
        
        % Adds a layer
        function obj = add(obj, layer)
            obj.layers{end + 1} = layer;
        end
        
        % Outputs prediction on each datum
        function prediction = predict(obj, x)
            prediction = {};
            for data_idx = 1:length(x)

                    data = x{data_idx};
                    
                    % Prediction is value after passed though all layers
                    for idx = 1:length(obj.layers)
                        [data, obj.layers{idx}] = obj.layers{idx}.forward(data);
                    end

                    prediction{end + 1} = data;
    
            end
        end
        
        % Function is for training the model 
        function model = train(obj, x_train, y_train, x_val, y_val, EPOCHS, LR, SCHEDULER)
            
            % Initialize Step_LR_Scheduler to adjust the learning rate
            scheduler = Step_LR_Scheduler(1000, 0.3, LR);
            train_acc = [];
            val_acc = [];
            
            % Occurs every full iteration (epoch)
            for epoch = 1:EPOCHS

                correct = 0;
                total = 0;
                total_loss = 0;

                for data_idx = 1:length(x_train)

                    data = x_train{data_idx};
%                     disp(length(data));
                                        
                    % Find prediction
                    for idx = 1:length(obj.layers)
                        [data, obj.layers{idx}] = obj.layers{idx}.forward(data);
%                          disp(["Layer: ", idx, data])
                    end
                    
                    % Keeps track of the total amount of loss 
                    total_loss = total_loss + obj.loss(y_train{data_idx}, data);
                    total = total + 1;
                    [~, ground_truth] = max(y_train{data_idx});
                    [~, pred_label] = max(data);
    
                    if ground_truth == pred_label
                        correct = correct + 1;
                    end

%                     disp(["Data", data, "Ground", y{data_idx}]);
%                     disp(y{data_idx})
%                     disp(obj.d_loss(y{data_idx}, data))
                  
                    cur_d_loss = obj.d_loss(y_train{data_idx}, data);
%                     disp(["Loss" cur_d_loss]);
                    
                    % Gradient Descent
                    for idx = length(obj.layers):-1:1
%                         disp(["LAYER# " idx])
                        % Backpropogation occurs based on the layer, the derivative's current loss, and learning rate
                        [cur_d_loss, obj.layers{idx}]= back(obj.layers{idx}, cur_d_loss, LR);
                    end

                    clc;
                    fprintf('Epoch: %i Training\n', epoch);
                    fprintf('Accuracy: %6.3f\n', correct / total * 100);
                    fprintf('Total loss: %10.3f\n', total_loss);
                       
                end

                train_acc = [train_acc, correct / total * 100];

                correct = 0;
                total = 0;

                for data_idx = 1:length(x_val)

                    data = x_val{data_idx};
%                     disp(length(data));
                                        
                    % Find prediction
                    for idx = 1:length(obj.layers)
                        [data, obj.layers{idx}] = obj.layers{idx}.forward(data);
%                          disp(["Layer: ", idx, data])
                    end
                    
                    [~, ground_truth] = max(y_val{data_idx});
                    [~, pred_label] = max(data);
                    total = total + 1;
                    if ground_truth == pred_label
                        correct = correct + 1;
                    end

                    clc;
                    fprintf('Epoch: %i Validation\n', epoch);
                    fprintf('Accuracy: %6.3f\n', correct / total * 100);
                       
                end

                val_acc = [val_acc, correct / total * 100];          

                % LR is based off of Step_LR_Scheduler's step function
                LR = scheduler.step();
            end

            epochs = 1:EPOCHS;

            figure(1);
            hold on;
            plot(epochs, train_acc);

            plot(epochs, val_acc);

            legend('Training Accuracy', 'Validation Accuracy');

            xlabel("Epoch");
            ylabel("Accuracy (%)");
            hold off;

            % Assigns the object to become the newest model
            model = obj;
            
        end
    end
end