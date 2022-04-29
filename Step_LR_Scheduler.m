% This is the Step Scheduler file which adjusts the learning rate for fine-tuned gradient descent
classdef Step_LR_Scheduler
    
    % This lists variables needed
    properties
        GAMMA
        STEP_SIZE
        LR
        idx
    end

    methods
    
        function obj = Step_LR_Scheduler(STEP_SIZE, GAMMA, LR)
            obj.STEP_SIZE = STEP_SIZE;
            obj.GAMMA = GAMMA;
            obj.idx = 0;
            obj.LR = LR;
        end
        
        % Adjusts the LR by GAMMA every STEP_SIZE steps
        function [LR, obj] = step(obj)
            obj.idx = obj.idx + 1;
            if mod(obj.idx, obj.STEP_SIZE) == obj.STEP_SIZE
                obj.LR = obj.LR * obj.GAMMA;
            end
            LR = obj.LR;
        end
    end

end