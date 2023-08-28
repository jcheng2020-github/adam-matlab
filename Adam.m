%Matlab-Python-Adam\Adam.m
%latest updated time: July 24, 2023
%Author: Junfu Cheng
%Email: jcheng2020@my.fit.edu
classdef Adam < handle
    properties
        model
        parameters
        gradient
        g
        loss

        lr
        betas
        weight_decay
        amsgrad
        maximize

        m
        v

        m_cor
        v_cor

        v_cor_max

        eps

        iteration
    end
    methods
        function obj = Adam(model, lr, betas, eps, weight_decay, amsgrad, maximize)
            switch nargin
                case 1
                    lr=0.001;
                    betas=[0.9, 0.999];
                    eps=1e-08;
                    weight_decay=0;
                    amsgrad=false;
                    maximize=false;
                    
                case 2
                    betas=[0.9, 0.999];
                    eps=1e-08;
                    weight_decay=0;
                    amsgrad=false;
                    maximize=false;
                    
                case 7
                otherwise
                    lr=0.001;
                    betas=[0.9, 0.999];
                    eps=1e-08;
                    weight_decay=0;
                    amsgrad=false;
                    maximize=false;
                    disp("Warning: model input required in Adam class construction");
                    disp("Calling setModel method of Adam is recommonded");
                    disp("--from Adam constructor");
            end
            obj.model = model;
            obj.lr = lr;
            obj.betas = betas;
            obj.eps = eps;
            obj.weight_decay = weight_decay;
            obj.amsgrad = amsgrad;
            obj.maximize = maximize;
        end
        function out = setlearningRate(obj, lr)
            arguments
                obj (1,1) Adam
                lr (1,1) double
            end
            obj.lr = lr;
            out = obj;
        end

        
        function out = getIteration(obj)
            out = obj.iteration;
        end



        function out = setModel(obj, model)
            obj.restart();
            obj.model = model;
            out = obj.model;
        end
        function out = restart(obj)
            obj.iteration = 1;%iteration begin from 1
            obj.intialize_ctrls();%obj.model.getParameters and update obj.parameters
            
            obj.m = zeros(1,length(obj.parameters));
            %first moment, an exponential average of the gradient
            obj.v = zeros(1,length(obj.parameters));
            %second moment, an exponential average of the square of the gradient
            obj.m_cor = zeros(1,length(obj.parameters));
            obj.v_cor = zeros(1,length(obj.parameters));
            obj.v_cor_max = zeros(1,length(obj.parameters));
            
            out = obj;
        end
        function out = intialize_ctrls(obj)
            obj.parameters = obj.model.getParameters();
            out = obj.parameters;
        end

        function out = evalGrad(obj)
            obj.model.setParameters(obj.parameters);
            obj.gradient = obj.model.getGradient();
            out = obj.gradient;
        end
        function out = evalObj(obj)
            obj.model.setParameters(obj.parameters);
            obj.loss = obj.model.getLoss();
            out = obj.loss;
        end

        function [loss, solution] = step(obj)
            obj.g = obj.evalGrad();%obj.model.setParameters and get gradient
            if obj.maximize == true
                obj.g = -obj.g;
            end

            if obj.weight_decay ~= 0
                obj.g = obj.g + obj.weight_decay * obj.parameters;
            end
            
            %Decay the first and second moment running average coefficient
            %(exponential average)
            obj.m = obj.betas(1) * obj.m + (1 - obj.betas(1))*obj.g;
            obj.v = obj.betas(2) * obj.v + (1 - obj.betas(2))*((obj.g).^2);


            bias_correction1 = 1 - obj.betas(1)^(obj.iteration);
            bias_correction2 = 1 - obj.betas(2)^(obj.iteration);

            obj.m_cor = obj.m/bias_correction1;
            obj.v_cor = obj.v/bias_correction2;

            if obj.amsgrad == true
                obj.v_cor_max = max(cat(1,obj.v_cor,obj.v_cor_max));
                obj.parameters = obj.parameters - obj.lr * ((obj.m_cor)./(sqrt(obj.v_cor_max) + obj.eps));
            else
                obj.parameters = obj.parameters - obj.lr * ((obj.m_cor)./(sqrt(obj.v_cor) + obj.eps));
            end


            obj.iteration = obj.iteration + 1;
            loss = obj.evalObj();%obj.model.setParameters and get Loss function value
            solution = obj.parameters;
        end

        function [loss, stepSize, solution] = GoldenSectionSearchStep(obj)
            obj.g = obj.evalGrad();%obj.model.setParameters and get gradient
            if obj.maximize == true
                obj.g = -obj.g;
            end

            if obj.weight_decay ~= 0
                obj.g = obj.g + obj.weight_decay * obj.parameters;
            end
            
            %Decay the first and second moment running average coefficient
            %(exponential average)
            obj.m = obj.betas(1) * obj.m + (1 - obj.betas(1))*obj.g;
            obj.v = obj.betas(2) * obj.v + (1 - obj.betas(2))*((obj.g).^2);


            bias_correction1 = 1 - obj.betas(1)^(obj.iteration);
            bias_correction2 = 1 - obj.betas(2)^(obj.iteration);

            obj.m_cor = obj.m/bias_correction1;
            obj.v_cor = obj.v/bias_correction2;
            
            currentSolution = obj.parameters;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ratio = (sqrt(5) - 1) / 2;
            e = 10^(-5);
            lr1 = 0;
            lr2 = obj.lr;
            
            length = lr2 - lr1;
            x1 = lr2 - length * ratio;
            x2 = lr1 + length * ratio;
            
            obj.parameters = currentSolution;
            obj.stepImplementation(x1);
            L1 = obj.evalObj();

            obj.parameters = currentSolution;
            obj.stepImplementation(x2);
            L2 = obj.evalObj();

            if L1 <= L2
                lr2 = x2;
                selectedLearningRate = x1;
            else
                lr1 = x1;
                selectedLearningRate = x2;
            end

            while abs(lr2 - lr1) > e
                length = lr2 - lr1;
                if L1 <= L2
                    x2 = x1;
                    x1 = lr2 - length * ratio;
                    
                    L2 = L1;
                    obj.parameters = currentSolution;
                    obj.stepImplementation(x1);
                    L1 = obj.evalObj();
                else
                    x1 = x2;
                    x2 = lr1 +length * ratio;

                    L1 = L2;
                    obj.parameters = currentSolution;
                    obj.stepImplementation(x2);
                    L2 = obj.evalObj();
                end

                if L1 <= L2
                    lr2 = x2;
                    selectedLearningRate = x1;
                else
                    lr1 = x1;
                    selectedLearningRate = x2;
                end
                fprintf("StepSize = %.20f\n",selectedLearningRate);
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.parameters = currentSolution;
            stepImplementation(obj, selectedLearningRate);


            obj.iteration = obj.iteration + 1;
            loss = obj.evalObj();%obj.model.setParameters and get Loss function value
            stepSize = selectedLearningRate;
            solution = obj.parameters;
        end
        function stepImplementation(obj, lr)
            if obj.amsgrad == true
                obj.v_cor_max = max(cat(1,obj.v_cor,obj.v_cor_max));
                obj.parameters = obj.parameters - lr * ((obj.m_cor)./(sqrt(obj.v_cor_max) + obj.eps));
            else
                obj.parameters = obj.parameters - lr * ((obj.m_cor)./(sqrt(obj.v_cor) + obj.eps));
            end
        end
    end
end