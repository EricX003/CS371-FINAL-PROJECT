classdef Cyclic_LR_Scheduler
 % This lists variables needed
    properties
        GAMMA
        STEP_SIZE
        LR
        idx
        MAX_LR
        BASE_LR
    end

    methods
    
        function obj = Cyclic_LR_Scheduler(STEP_SIZE, GAMMA, BASE_LR, MAX_LR)
            obj.STEP_SIZE = STEP_SIZE;
            obj.GAMMA = GAMMA;
            obj.idx = 0;
            obj.LR = BASE_LR;
            obj.MAX_LR = MAX_LR - BASE_LR;
            obj.BASE_LR = BASE_LR;
        end
        
        % Adjusts the LR by GAMMA every STEP_SIZE steps
        function [LR, obj] = step(obj)
            obj.idx = obj.idx + 1;
            if mod(obj.idx, obj.STEP_SIZE) == obj.STEP_SIZE
                if mod(obj.idx / obj.STEP_SIZE, 2) == 1
                    obj.LR = obj.BASE_LR + obj.MAX_LR * obj.GAMMA ^ ((obj.idx / obj.STEP_SIZE) + 1) / 2;
                else
                    obj.LR = obj.BASE_LR;
                end
            end
            LR = obj.LR;
        end
    end
end