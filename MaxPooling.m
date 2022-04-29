classdef MaxPooling

    properties
        PoolingSize
        in
    end

    methods(Static = false)
        function obj = MaxPooling(width)
            obj.PoolingSize = width;
        end
        
        % Function to perform forward propagation 
        function [obj, out] = forward(obj, in)

            obj.in = in;

            [height, width, depth] = size(in);
            out = zeros([height / 2, width / 2, depth]);

            for x = 0 : height / obj.PoolingSize - 1
                for y = 0 : width / obj.PoolingSize - 1
                    out(x + 1, y + 1, :) = max(in(x * obj.PoolingSize + 1: x * obj.PoolingSize + obj.PoolingSize, y * obj.PoolingSize + 1: y * obj.PoolingSize + obj.PoolingSize, : ), [], [1, 2]);
                end
            end
        end
        
        % Function to perform backpropagation 
        function [obj, dLdIn] = back(obj, dLdOut, LR)

            dLdIn = zeros(size(obj.in));

            [height, width, depth] = size(obj.in);

            for x = 1 : height / obj.PoolingSize
                for y = 1: width / obj.PoolingSize
                   sub = obj.in((x - 1) * obj.PoolingSize + 1: x * obj.PoolingSize, (y - 1) * obj.PoolingSize + 1: y * obj.PoolingSize, : );
                   max_val = max(sub, [], [1, 2]);
                   [A, B, C] = size(sub);

                   for a = 1:A
                       for b = 1:B
                           for c = 1:C
                               if sub(a, b, c) == max_val(c)
                                   dLdIn(x * 2 + a, y * 2 + b, c) = dLdOut(x, y, c);
                               end
                           end
                       end
                   end
                end
            end
        end

    end

end
