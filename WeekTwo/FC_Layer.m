classdef FC_Layer

    properties
        weight
        bias
        in
        out
    end

    methods(Static = false)

        function obj = FC_Layer(in, out) 
            obj.weight = rand(in, out) - 0.5;
            obj.bias = rand(1, out) - 0.5;
        end

        function [output, obj] = forward(obj, in)
            obj.in = in;
            obj.out = obj.in * obj.weight + obj.bias;
            output = obj.out;
        end
        
        function [dEdX, obj] = back(obj, loss, LR)

%             disp(["OBJ WEIGHT" size(obj.weight)])
%             disp(["OBJ IN" size(obj.in)])
%             disp(["LOSS" size(loss)])

            obj.weight = obj.weight - LR * obj.in' * loss;
            obj.bias = obj.bias - LR * loss;

            dEdX = loss * obj.weight';
        end

    end

end