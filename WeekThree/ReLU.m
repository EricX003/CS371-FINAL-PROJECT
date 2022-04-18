classdef ReLU

    properties
        in
        out
        act
        d_act
    end

    methods (Static = false)
        function obj = ReLU()
             obj.act = @(x) tanh(x);a
             obj.d_act = @(x) 1-tanh(x).^2;
        end

        function [output, obj] = forward(obj, in)
            obj.in = in;
            obj.out = obj.act(obj.in);
            output = obj.out;
        end

        function [dEdX, obj] = back(obj, loss, LR)
%             disp(["IN" obj.in])
%             disp(["d act" obj.d_act(obj.in)]);

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