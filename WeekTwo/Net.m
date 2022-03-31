% The Net is where the model is trained and all the layers are added
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
        
        % Outputs prediction which causes the network to adjust its weights based off of this prediction to achieve more accuracy
        function prediction = predict(obj, x)
            prediction = {};
            for data_idx = 1:length(x)

                    data = x{data_idx};
                    
                    % Bases prediction off of the last layers results
                    for idx = 1:length(obj.layers)
                        [data, obj.layers{idx}] = obj.layers{idx}.forward(data);
                    end

                    prediction{end + 1} = data;
    
            end
        end
        
        % Function is for training the model 
        function model = train(obj, x, y, EPOCHS, LR)
            
            % Calls upong the Step_LR_Scheduler to adjust the learning rate
            scheduler = Step_LR_Scheduler(1000, 0.3, LR);

            last = -1;
            
            % Occurs every full iteration (epoch)
            for epoch = 1:EPOCHS
                %disp("Epoch")
                total_loss = 0;
                for data_idx = 1:length(x)

                    data = x{data_idx};
                    
                    % Stores the data
                    for idx = 1:length(obj.layers)
                        [data, obj.layers{idx}] = obj.layers{idx}.forward(data);
                    end
                    
                    % Keeps track of the total ammount of loss 
                    total_loss = total_loss + obj.loss(y{data_idx}, data);
                    
                    cur_d_loss = obj.d_loss(y{data_idx}, data);
%                     disp(["Loss" cur_d_loss]);
                    
                    %Runs through the layers backwards
                    for idx = length(obj.layers):-1:1
%                         disp(["LAYER# " idx])
                        % Backpropogation occurs based on the layer, the derivative's current loss, and learn rate
                        [cur_d_loss, obj.layers{idx}]= back(obj.layers{idx}, cur_d_loss, LR);
                    end
    
                end
                
                disp(total_loss)
                
                if last == total_loss
                    disp
                end
                last = total_loss;
                
                % LR is based off of Step_LR_Scheduler's step function
                LR = scheduler.step();
            end
            
            % Assigns the object to become the newest model
            model = obj;
            
        end
    end
end
