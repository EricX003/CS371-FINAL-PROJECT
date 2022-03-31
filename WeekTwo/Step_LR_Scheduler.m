% This is the Step Scheduler file which adjusts the learning rate for more accuracy 
classdef Step_LR_Scheduler
    
    % This lists variables needed
    properties
        GAMMA
        STEP_SIZE
        LR
        idx
    end

    methods
        % This has all the original values inputted in
        function obj = Step_LR_Scheduler(STEP_SIZE, GAMMA, LR)
            obj.STEP_SIZE = STEP_SIZE;
            obj.GAMMA = GAMMA;
            obj.idx = 0;
            obj.LR = LR;
        end
        
        % After every step size (ie. if stepsize is 3 then after every 3 iterations) it decreases the learning rate 
        function LR = step(obj)
            obj.idx = obj.idx + 1;
            if mod(obj.idx, obj.STEP_SIZE) == obj.STEP_SIZE
                obj.LR = obj.LR * obj.GAMMA;
            end
            LR = obj.LR;
        end
    end

end
