 % Has capactiy to apply activation function to input and backpropogate the derivative of the activation function during gradient descent.
classdef Activation_Layer

    % All values needed
    properties
        in
        out
        act
        d_act
    end

    % Non-static methods
    methods (Static = false)
    
        % Based on user input, the layer's activation finction is set
        function obj = Activation_Layer(type)
             % Switch case that detects your activation function choice, set the function and its derivative
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
                     
                     % Default function
                 otherwise
                     obj.act = @(x) obj.ReLU(x);
                     obj.d_act = @(x) obj.d_ReLU(x);
             end

        end

        % Based off of chosen function, applies activation function to teh data
        function [output, obj] = forward(obj, in)
            obj.in = in;
            obj.out = obj.act(obj.in);
            output = obj.out;
        end
        
        % Backpropogation that modifies the loss in order to fit FC layers
        function [dEdX, obj] = back(obj, loss, LR)
            dEdX = obj.d_act(obj.in) .* loss;
        end

    end
    
    % Static Functions
    methods(Static = true)
        
        function x = ReLU(x)
            x(x < 0) = 0;
        end
        
        function d_x = d_ReLU(x)
            x(x > 0) = 1;
            x(x < 0) = 0;
            d_x = x;
        end
        
        function x = Leaky_ReLU(x)
            x(x < 0) = 0.1 * x(x < 0);
        end
        
        function d_x = d_Leaky_ReLU(x)
            x(x > 0) = 1;
            x(x < 0) = 0.1;
            d_x = x;
        end

    end

end
