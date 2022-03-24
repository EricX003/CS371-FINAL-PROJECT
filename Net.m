classdef Net

    properties
        layers
        loss
        d_loss

    end

    methods

        function obj = Net(loss, d_loss)
            obj.layers = {};
            obj.loss = loss;
            obj.d_loss = d_loss;
        end
    
        function layers = add(obj, layer)
            disp("ADDING LAYER")
            disp(length(obj.layers))
            obj.layers{end + 1} = layer;
            layers = obj.layers;
        end
    
        function prediction = predict(obj, input_data)
            samples = length(input_data);
            result = [];
            
            for i = 1:samples
                output = input_data(i);
                for layer = obj.layers;
                    output = layer.forward_propagation(output);
                end
                result = [result output];
            end
            
            prediction = result;
        end
    
        function train(obj, x, y, EPOCHS, LR)

            for epoch = 1:EPOCHS
                %disp("epoch")
                total_loss = 0;
                for data_idx = 1:length(x)
                    %disp('data')
                    data = x(data_idx);

                    for layer = obj.layers
                        data = layer.forward(data);
                    end

                    total_loss = total_loss + obj.loss(y(data_idx), data);
                    
                    cur_d_loss = obj.d_loss(y(data_idx), data);

                    
                    for layer = length(obj.layers):-1:1

                        cur_d_loss = obj.layers{layer}.backward(cur_d_loss, LR);
                        %disp('going back')
                    end
    
                end
                %disp(total_loss);
            end

            
        end
    end
end
