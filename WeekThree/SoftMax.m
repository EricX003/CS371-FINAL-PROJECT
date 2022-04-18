classdef SoftMax

    properties
        weight
        bias
        in
        out
    end

    methods(Static = false)
        
        function obj = SoftMax(in, out)

            obj.in = in;
            obj.out = out;
            obj.weight - randn(in, out) - 0.5;
            obj.bias = randn(out) - 0.5;

        end

        function out = forward(obj, in)
            in = reshape(in, [1, prod(size(in))]);

            e_power = exp(in .* obj.weight + obj.bias);
            out = e_power / sum(e_power)
        end

    end
end