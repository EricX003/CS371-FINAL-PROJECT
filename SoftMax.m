classdef SoftMax

    properties
        weight
        bias
        in_size
        out_size
        in
        e_power
        last_size
    end

    methods(Static = false)
        
        function obj = SoftMax(in, out)

            obj.in_size = in;
            obj.out_size = out;
            obj.weight = randn(in, out) - 0.5;
            obj.bias = randn(1, out) - 0.5;

        end

        function [obj, out] = forward(obj, in)

            obj.last_size = size(in);
            obj.in = reshape(in, [1, numel(in)]);

%             disp(obj.in_size);
%             disp(obj.out_size);
%             disp(size(obj.in))

            obj.e_power = exp(obj.in * obj.weight + obj.bias);
%             disp(obj.e_power);

            out = obj.e_power ./ sum(obj.e_power);
        end

        function [obj, out] = back(obj, dL, LR)
            for idx = 1:length(dL)
                grad = dL(idx);

                if grad ~= 0

%                     disp(size(obj.e_power))

                    sum_e = sum(obj.e_power);

                    dOut = -obj.e_power(idx) .* obj.e_power ./ sum_e ^ 2;
                    dOut(idx) = obj.e_power(idx) * (sum_e - obj.e_power(idx)) / sum_e ^ 2;

                    dLdW = [obj.in]' * [grad .* dOut];
                    dLdB = grad * dOut;

%                     disp(size(obj.weight));
%                     disp(size(grad .* dOut));

                    dLdIN = obj.weight * (grad .* dOut)';

                    obj.weight = obj.weight - LR * dLdW;
                    obj.bias = obj.bias - LR * dLdB;

                    out = reshape(dLdIN, obj.last_size);
                end
            end
        end

    end
end