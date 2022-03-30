% Activation Layer 
classdef Activation_Layer

    % Lists all values needed for the file
    properties
        in
        out
        act
        d_act
    end

    % Sets methods below to non-static
    methods (Static = false)
    
        % Based on user input and shifts the function type
        function obj = Activation_Layer(type)
             switch type
                 case 'tanh'
                     obj.act = @(x) tanh(x);
                     obj.d_act = @(x) 1 - tanh(x) .^ 2;
                 case 'leaky_relu'
                     obj.act = @(x) obj.Leaky_ReLU(x);
                     obj.d_act = @(x) obj.d_Leaky_ReLU(x);
                 case 'sigmoid'
                     obj.act = @(x) 1 ./ (1 + exp(-x));
                     obj.d_act = @(x) exp(-x) ./ (x + exp(-x)) .^ 2;
                 otherwise
                     obj.act = @(x) obj.ReLU(x);
                     obj.d_act = @(x) obj.d_ReLU(x);
             end

        end

        % Based off of their chosen function, pushes the data forward with the calculated weight and bias
        function [output, obj] = forward(obj, in)
            obj.in = in;
            obj.out = obj.act(obj.in);
            output = obj.out;
        end
        
        % Backpropogation that readjusts based on their chosen function, therefore lowering the Loss 
        function [dEdX, obj] = back(obj, loss, LR)
            dEdX = obj.d_act(obj.in) .* loss;
        end

    end
    
    % Sets any methods below to static methods
    methods(Static = true)
        
        % ReLU(x) = max(0, x), doesn't let anything be negative
        function x = ReLU(x)
            x(x < 0) = 0;
        end
        
        % This function sets negative numbers to 0 and positive to 1 for the slope
        function d_x = d_ReLU(x)
            x(x > 0) = 1;
            x(x < 0) = 0;
            d_x = x;
        end
        
        % Allows some negatives not all 
        function x = Leaky_ReLU(x)
            x(x < 0) = 0.1 * x(x < 0);
        end
        
        % Allows some negatives but not all 
        function d_x = d_Leaky_ReLU(x)
            x(x > 0) = 1;
            x(x < 0) = 0.1;
            d_x = x;
        end

    end

end
