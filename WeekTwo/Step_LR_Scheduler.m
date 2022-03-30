classdef Step_LR_Scheduler

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

        function LR = step(obj)
            obj.idx = obj.idx + 1;
            if mod(obj.idx, obj.STEP_SIZE) == obj.STEP_SIZE
                obj.LR = obj.LR * obj.GAMMA;
            end
            LR = obj.LR;
        end
    end

end