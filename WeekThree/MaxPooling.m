classdef MaxPooling

    properties
        PoolingSize
    end

    methods(Static = false)
        function obj = MaxPooling(width)
            obj.PoolingSize = width;
        end

        function out = forward(obj, in)
            
            [height, width, depth] = size(in);
            out = zero(height / 2, width / 2, depth);

            for x = 1 : height / obj.PoolingSize
                for y = 1: width / obj.PoolingSize
                    out(x, y) = max(in(x * obj.PoolingSize : x * obj.PoolingSize + obj.PoolingSize - 1, y * obj.PoolingSize : y * obj.PoolingSize + obj.PoolingSize - 1, : ), [], [1, 2]);
                end
            end
        end

    end

end