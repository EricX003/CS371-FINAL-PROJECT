classdef Convolutional_Layer

    properties
        depth
        kernel
        kernel_size
        in
    end

    methods(Static = false)
        
        % Function initializes the kernel according to the depth and size
        function obj = Convolutional_Layer(depth, size)
            obj.kernel_size = size;
            obj.kernel = randn(depth, obj.kernel_size, obj.kernel_size) / 9;
            obj.depth = depth;
        end
        
        % Function to perform forward propogation 
        function [obj, out] = forward(obj, in)
            obj.in = in;
            [height, width] = size(in);
            out = zeros(height - obj.kernel_size + 1, width - obj.kernel_size + 1, obj.depth);
   
            for x = 1:height - obj.kernel_size + 1
                for y = 1: width - obj.kernel_size + 1
                
                    % Performs a dot product between the input and kernel
                    out(x, y) = sum(in(x : x + obj.kernel_size - 1, y : y + obj.kernel_size - 1) .* obj.kernel, ["all"]);
                end
            end

        end
        
        % Function to perform backpropagation
        function [obj, dLdF] = back(obj, dLdOut, LR)

            dLdF = zeros(size(obj.kernel));
            [height, width] = size(obj.in);

            for x = 1:height - obj.kernel_size + 1
                for y = 1: width - obj.kernel_size + 1
                    for d = 1:obj.depth
                        dLdF(d, :, :) = dLdOut(x, y, d) * obj.in(x : x + obj.kernel_size - 1, y : y + obj.kernel_size - 1);
                    end
                end
            end

            obj.kernel = obj.kernel - LR * dLdF;

        end
    end

end
