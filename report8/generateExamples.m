function dataset = generateExamples(datasetPath, config)
%GENERATEEXAMPLES Generate and save independent synthetic dataset splits.
% Run from Octave with generateExamples(). An optional config structure can
% override sizes, counts, and random seed.

scriptDirectory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(datasetPath)
    datasetPath = fullfile(scriptDirectory, 'examples.mat');
end
if nargin < 2 || isempty(config)
    config = struct();
end
if ~ischar(datasetPath) || isempty(datasetPath)
    error('datasetPath must be a non-empty character vector.');
end

config = setDefault(config, 'inputSize', 16);
config = setDefault(config, 'surfaceResolution', 16);
config = setDefault(config, 'trainingExamples', 510);
config = setDefault(config, 'validationExamples', 85);
config = setDefault(config, 'testExamples', 85);
config = setDefault(config, 'seed', 2026);
validateConfig(config);

fprintf('Generating %d training, %d validation, and %d test examples...\n', ...
    config.trainingExamples, config.validationExamples, config.testExamples);
[trainingInputs, trainingTargets] = generateSyntheticData( ...
    config.trainingExamples, config.inputSize, config.surfaceResolution, config.seed + 1);
[validationInputs, validationTargets] = generateSyntheticData( ...
    config.validationExamples, config.inputSize, config.surfaceResolution, config.seed + 2);
[testInputs, testTargets] = generateSyntheticData( ...
    config.testExamples, config.inputSize, config.surfaceResolution, config.seed + 3);

dataset = struct();
dataset.config = config;
dataset.trainingInputs = trainingInputs;
dataset.trainingTargets = trainingTargets;
dataset.validationInputs = validationInputs;
dataset.validationTargets = validationTargets;
dataset.testInputs = testInputs;
dataset.testTargets = testTargets;

outputDirectory = fileparts(datasetPath);
if ~isempty(outputDirectory) && exist(outputDirectory, 'dir') ~= 7
    [created, message] = mkdir(outputDirectory);
    if ~created
        error('Could not create dataset directory: %s', message);
    end
end
save(datasetPath, '-struct', 'dataset', '-mat7-binary');
fprintf('Saved synthetic examples to %s\n', datasetPath);
end

function config = setDefault(config, fieldName, defaultValue)
if ~isstruct(config) || ~isscalar(config)
    error('config must be a scalar structure.');
end
if ~isfield(config, fieldName) || isempty(config.(fieldName))
    config.(fieldName) = defaultValue;
end
end

function validateConfig(config)
integerFields = {'inputSize', 'surfaceResolution', 'trainingExamples', ...
    'validationExamples', 'testExamples', 'seed'};
for fieldIndex = 1:numel(integerFields)
    fieldName = integerFields{fieldIndex};
    value = config.(fieldName);
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value ~= floor(value)
        error('config.%s must be a finite integer.', fieldName);
    end
end
if config.inputSize < 2 || config.surfaceResolution < 2 || ...
        config.trainingExamples < 1 || config.validationExamples < 1 || ...
        config.testExamples < 1
    error('Image/surface resolutions and split counts must be positive and valid.');
end
end