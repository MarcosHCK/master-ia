% INFER_IMAGE.M Reconstruct a surface from a top-down depthmap checkpoint.
% Set imagePath below to a depthmap image, then run:
%   octave --no-gui --quiet infer_image.m
% Or call inferImage(weightsPath, imagePath, outputDirectory) directly.

scriptDirectory = fileparts(mfilename('fullpath'));
weightsPath = fullfile(scriptDirectory, 'trained-network.mat');
imagePath = fullfile(scriptDirectory, 'input.png');
outputDirectory = scriptDirectory;

inferenceResult = inferImage(weightsPath, imagePath, outputDirectory);