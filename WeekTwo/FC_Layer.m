% This file is the Fully-Connected layer which runs the data through and affects it with the weights and bias.
classdef FC_Layer

    % Listing all the needed properties
    properties
        weight
        bias
        in
        out
    end

    % Declaring method non-static
    methods(Static = false)

        % This function calculates the weight and bias used in the forward method
        function obj = FC_Layer(in, out) 
            obj.weight = rand(in, out) - 0.5;
            obj.bias = rand(1, out) - 0.5;
        end

        % This function moves the data forward and calculates it with the weight and bias before pushing it 'out' (returning the value)
        function [output, obj] = forward(obj, in)
            obj.in = in;
            obj.out = obj.in * obj.weight + obj.bias;
            output = obj.out;
        end
        
        % This function is for backpropogation and readjusts the weight and bias, therefore reducing the loss
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
