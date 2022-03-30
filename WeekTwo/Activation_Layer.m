classdef Activation_Layer

    properties
        in
        out
        act
        d_act
    end

    methods (Static = false)
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

        function [output, obj] = forward(obj, in)
            obj.in = in;
            obj.out = obj.act(obj.in);
            output = obj.out;
        end

        function [dEdX, obj] = back(obj, loss, LR)
            dEdX = obj.d_act(obj.in) .* loss;
        end

    end

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