classdef Convolutional_Layer

    properties
        depth
        kernel
        width
    end

    methods(Static = false)

        function obj = Convolutional_Layer(depth, size)
            obj.width = size;
            obj.kernel = randn(depth, obj.width, obj.width) / 9;
            obj.depth = depth;
        end

        function out = forward(obj, in)
            [height, width] = obj.width(in);
            out = zeros(height - obj.width - 1, width - obj.width - 1, depth);

            for x = 1:height - obj.width - 1
                for y = 1: width - obj.width - 1
                    out(x, y) = sum(in(x : x + obj.width, y : y + obj.width) .* kernel, [1, 2])
                end
            end

        end
    end

end