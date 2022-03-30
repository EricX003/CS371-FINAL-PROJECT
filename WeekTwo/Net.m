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
    
        function obj = add(obj, layer)
            obj.layers{end + 1} = layer;
        end
    
        function prediction = predict(obj, x)
            prediction = {};
            for data_idx = 1:length(x)

                    data = x{data_idx};

                    for idx = 1:length(obj.layers)
                        [data, obj.layers{idx}] = obj.layers{idx}.forward(data);
                    end

                    prediction{end + 1} = data;
    
            end
        end
    
        function model = train(obj, x, y, EPOCHS, LR)

            scheduler = Step_LR_Scheduler(1000, 0.3, LR);

            last = -1;

            for epoch = 1:EPOCHS
                %disp("Epoch")
                total_loss = 0;
                for data_idx = 1:length(x)

                    data = x{data_idx};

                    for idx = 1:length(obj.layers)
                        [data, obj.layers{idx}] = obj.layers{idx}.forward(data);
                    end

                    total_loss = total_loss + obj.loss(y{data_idx}, data);
                    
                    cur_d_loss = obj.d_loss(y{data_idx}, data);
%                     disp(["Loss" cur_d_loss]);

                    for idx = length(obj.layers):-1:1
%                         disp(["LAYER# " idx])
                        [cur_d_loss, obj.layers{idx}]= back(obj.layers{idx}, cur_d_loss, LR);
                    end
    
                end
                disp(total_loss)
                if last == total_loss
                    disp
                end
                last = total_loss;
                LR = scheduler.step();
            end
            model = obj;
            
        end
    end
end
