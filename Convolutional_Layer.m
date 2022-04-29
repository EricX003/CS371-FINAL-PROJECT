classdef Convolutional_Layer

    properties
        depth
        kernel
        kernel_size
        in
    end

    methods(Static = false)

        function obj = Convolutional_Layer(size, depth)
            obj.kernel_size = size;
            %We empirically found
            obj.kernel = randn(obj.kernel_size, obj.kernel_size, depth) / 9;
            obj.depth = depth;
        end

        function [obj, out] = forward(obj, in)
            obj.in = in;
            [height, width] = size(in);
            out = zeros(height - obj.kernel_size + 1, width - obj.kernel_size + 1, obj.depth);

            for x = 1:height - obj.kernel_size + 1
                for y = 1: width - obj.kernel_size + 1
                    out(x, y, :) = sum(in(x : x + obj.kernel_size - 1, y : y + obj.kernel_size - 1) .* obj.kernel, [1:2]);
                end
            end

        end

        function [obj, dLdF] = back(obj, dLdOut, LR)

            dLdF = zeros(size(obj.kernel));
            [height, width] = size(obj.in);

            for x = 1:height - obj.kernel_size + 1
                for y = 1: width - obj.kernel_size + 1
                    for d = 1:obj.depth
                        dLdF(:, :, d) = dLdF(:, :, d) + dLdOut(x, y, d) * obj.in(x : x + obj.kernel_size - 1, y : y + obj.kernel_size - 1);
                    end
                end
            end

            obj.kernel = obj.kernel - LR * dLdF;

        end
    end

end