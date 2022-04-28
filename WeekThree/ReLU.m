classdef ReLU

    properties
        in
        out
        act
        d_act
    end

    methods (Static = false)
        % The activation layer's ReLU function and derivative
        function obj = ReLU()
             obj.act = @(x) tanh(x);a
             obj.d_act = @(x) 1-tanh(x).^2;
        end
        
        % Function to perform forward propogation
        function [output, obj] = forward(obj, in)
            obj.in = in;
            % Applies the function to the input
            obj.out = obj.act(obj.in);
            output = obj.out;
        end
        
        % Function to perform backpropagation
        function [dEdX, obj] = back(obj, loss, LR)
%             disp(["IN" obj.in])
%             disp(["d act" obj.d_act(obj.in)]);
            
            % Applies the derivative of the function to the input and the loss
            dEdX = obj.d_act(obj.in) .* loss;
        end

    end

%     methods(Static = true)
%         function res = act(x)
%             res = tanh(x);
%         end
% 
%         function res = d_act(x)
%             res = 1-tanh(x).^2;
%         end
%     end

end
