classdef FC_Layer

    properties
        weight;
        bias;
        input
        out
    end

    methods
        function obj = FC_Layer(in, out) 
            obj.weight = rand(in, out) - 0.5;
            obj.bias = rand(1, out) - 0.5;
        end

        function output = forward(obj,in)
            obj.input = in;
            obj.out = dot(obj.input, obj.weight) + obj.bias;
            output = obj.out;
        end

        function dEdX = back(obj, loss, LR)
            obj.weight = self.weight - LR * dot(obj.input', loss);
            obj.bias = obj.bias - LR * loss;

            dEdX = dot(loss, obj.weight');
            disp(obj.weight);
        end
    end
end
