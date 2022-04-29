classdef CNN

    properties
        layers
        NUM_CLASSES
    end

    methods(Static = false)

        function obj = CNN(num_classes)
            obj.layers = {};
            obj.NUM_CLASSES = num_classes;
        end

        function obj = add(obj, layer)
            obj.layers{end + 1} = layer;
        end

        function obj = train(obj, x, y, EPOCHS, LR)
            for epoch = 1:EPOCHS
                for data_idx = 1 : length(x)

                    [dummy, ground_truth] = max(y{data_idx});

                    data = x{data_idx};
    
                    for layer_idx = 1 : length(obj.layers)
                        [obj.layers{layer_idx}, data] = obj.layers{layer_idx}.forward(data);
                    end
    
                    [confidence, pred_label] = max(data);
    
                    disp(["LOSS", -log(data(ground_truth))]);
    
                    grad = zeros([1, obj.NUM_CLASSES]);
                    grad(ground_truth) = -1 / data(ground_truth);
    
                    for layer_idx = length(obj.layers) : -1 : 1
                        [obj.layers{layer_idx}, grad] = obj.layers{layer_idx}.back(grad, LR);
                    end
    
                end
            end
        end
    end

end