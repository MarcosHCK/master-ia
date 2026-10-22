% TRAIN_NETWORK.M Train from examples.mat and save trained-network.mat.
% First run generate_examples.m, then run:
%   octave --no-gui --quiet train_network.m

scriptDirectory = fileparts(mfilename('fullpath'));
datasetPath = fullfile(scriptDirectory, 'examples.mat');
weightsPath = fullfile(scriptDirectory, 'trained-network.mat');

trainingOptions.hiddenUnits = 96;
trainingOptions.learningRate = 0.003;
trainingOptions.maxEpochs = 3000;
trainingOptions.batchSize = 32;
trainingOptions.patience = 30;
trainingOptions.minImprovement = 1e-4;
trainingOptions.weightDecay = 1e-5;
trainingOptions.maxGradientNorm = 10;
trainingOptions.positiveClassWeight = 6;
trainingOptions.seed = 2026;
trainingOptions.exportEvaluationArtifacts = true;

trainModel(datasetPath, weightsPath, trainingOptions);