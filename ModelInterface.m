%Matlab-Python-Adam\ModelInterface.m
%latest updated time: July 24, 2023
%Author: Junfu Cheng
%Email: jcheng2020@my.fit.edu
classdef ModelInterface < handle
    properties(Abstract)
        parameters
        gradient
        loss
    end
    methods(Abstract)
        out = getParameters(obj)
        out = getGradient(obj)
        out = setParameters(obj, params)
        out = getLoss(obj)
    end
end