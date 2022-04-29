% Fully connected layer implementation
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

        % Random initialization of weight and bias to be a normal distribution between -0.5 and 0.5
        function obj = FC_Layer(in, out) 

            obj.weight = rand(in, out) - 0.5;
            obj.bias = rand(1, out) - 0.5;
        end

        % This function applies the weight and adds the bias to producte the layer's output
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
            
            % Updates weights during gradient descent
            obj.weight = obj.weight - LR * obj.in' * loss;
            obj.bias = obj.bias - LR * loss;

            dEdX = loss * obj.weight';
        end

    end

end