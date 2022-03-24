classdef ReLU

    properties
        in
        out
        act
        d_act
    end

    methods
        function obj = ReLU()
            obj.act = @(x) tanh(x);
            obj.d_act = @(x) 1-tanh(x)^2;
        end

        function output = forward(obj, in)
            obj.in = in;
            obj.out = act(in);
            output = obj.out;
        end

        function dEdX = back(obj, loss, LR)
            dEdX = obj.d_act(obj.in) * loss;
        end
            

    end

end