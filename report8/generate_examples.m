% GENERATE_EXAMPLES.M Generate and save training/validation/test pairs.
% Run once before train_network.m. Re-running regenerates the dataset.
%   octave --no-gui --quiet generate_examples.m

scriptDirectory = fileparts(mfilename('fullpath'));
datasetPath = fullfile(scriptDirectory, 'examples.mat');

datasetConfig.inputSize = 16;
datasetConfig.voxelResolution = 16;
datasetConfig.trainingExamples = 510;
datasetConfig.validationExamples = 85;
datasetConfig.testExamples = 85;
datasetConfig.seed = 2026;

generateExamples(datasetPath, datasetConfig);