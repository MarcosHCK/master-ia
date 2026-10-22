function [network, trainingSummary] = trainModel(datasetPath, weightsPath, options)
%TRAINMODEL Train from a saved dataset and write an inference-only checkpoint.
% Run from Octave with trainModel(). No examples are generated here.

scriptDirectory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(datasetPath)
    datasetPath = fullfile(scriptDirectory, 'examples.mat');
end
if nargin < 2 || isempty(weightsPath)
    weightsPath = fullfile(scriptDirectory, 'trained-network.mat');
end
if nargin < 3 || isempty(options)
    options = struct();
end
if ~ischar(datasetPath) || isempty(datasetPath) || ...
        ~ischar(weightsPath) || isempty(weightsPath)
    error('datasetPath and weightsPath must be non-empty character vectors.');
end

options = setDefault(options, 'hiddenUnits', 96);
options = setDefault(options, 'learningRate', 0.003);
options = setDefault(options, 'maxEpochs', 3000);
options = setDefault(options, 'batchSize', 32);
options = setDefault(options, 'patience', 30);
options = setDefault(options, 'minImprovement', 1e-4);
options = setDefault(options, 'weightDecay', 1e-5);
options = setDefault(options, 'maxGradientNorm', 10);
options = setDefault(options, 'positiveClassWeight', 6);
options = setDefault(options, 'seed', 2026);
options = setDefault(options, 'exportEvaluationArtifacts', true);
validateOptions(options);

if exist(datasetPath, 'file') ~= 2
    error('Dataset file does not exist: %s. Run generateExamples first.', datasetPath);
end
dataset = load(datasetPath);
requiredFields = {'config', 'trainingInputs', 'trainingTargets', ...
    'validationInputs', 'validationTargets', 'testInputs', 'testTargets'};
if ~all(isfield(dataset, requiredFields))
    error('Dataset file is missing one or more required splits or its config.');
end
config = dataset.config;
validateDatasetSplits(dataset, config);

network = buildNetwork(config.inputSize, config.voxelResolution, ...
    options.hiddenUnits, mean(dataset.trainingTargets(:)), options.seed);
network.occupancyThreshold = 0.5;
trainingOptions.learningRate = options.learningRate;
trainingOptions.maxEpochs = options.maxEpochs;
trainingOptions.batchSize = options.batchSize;
trainingOptions.patience = options.patience;
trainingOptions.minImprovement = options.minImprovement;
trainingOptions.weightDecay = options.weightDecay;
trainingOptions.maxGradientNorm = options.maxGradientNorm;
trainingOptions.positiveClassWeight = options.positiveClassWeight;
trainingOptions.seed = options.seed + 4;

fprintf('Training %d -> %d -> %d image-to-voxel network.\n', ...
    config.inputSize ^ 2, options.hiddenUnits, config.voxelResolution ^ 3);
network = trainNetwork(network, dataset.trainingInputs, dataset.trainingTargets, ...
    dataset.validationInputs, dataset.validationTargets, trainingOptions);
[network.occupancyThreshold, validationVoxelIoU] = ...
    selectOccupancyThreshold(network, dataset.validationInputs, dataset.validationTargets);
[testMetrics, testProbabilities] = evaluateNetwork(network, ...
    dataset.testInputs, dataset.testTargets);

inferenceConfig = struct('inputSize', network.inputSize, ...
    'voxelResolution', network.voxelResolution, ...
    'occupancyThreshold', network.occupancyThreshold);
trainingSummary = struct();
trainingSummary.datasetPath = datasetPath;
trainingSummary.datasetConfig = config;
trainingSummary.trainingOptions = options;
trainingSummary.trainingExamples = size(dataset.trainingInputs, 1);
trainingSummary.validationExamples = size(dataset.validationInputs, 1);
trainingSummary.testExamples = size(dataset.testInputs, 1);
trainingSummary.epochs = network.epochs;
trainingSummary.bestEpoch = network.bestEpoch;
trainingSummary.stoppedEarly = network.stoppedEarly;
trainingSummary.trainingLoss = network.trainingLoss;
trainingSummary.validationLoss = network.validationLoss;
trainingSummary.validationVoxelIoU = validationVoxelIoU;
trainingSummary.testMetrics = testMetrics;
trainingSummary.trainingHistory = struct( ...
    'epoch', network.epochHistory, ...
    'trainingBCE', network.trainingLossHistory, ...
    'validationBCE', network.validationLossHistory);

weightsDirectory = fileparts(weightsPath);
if ~isempty(weightsDirectory) && exist(weightsDirectory, 'dir') ~= 7
    [created, message] = mkdir(weightsDirectory);
    if ~created
        error('Could not create weights directory: %s', message);
    end
end
fprintf('Finished at epoch %d (best epoch %d; early stopping: %s).\n', ...
    network.epochs, network.bestEpoch, yesNo(network.stoppedEarly));
fprintf('Validation IoU %.4f; test voxel IoU %.4f; silhouette IoU %.4f.\n', ...
    validationVoxelIoU, testMetrics.voxelIoU, testMetrics.silhouetteIoU);

network = rmfield(network, {'trainingLoss', 'validationLoss', ...
    'trainingLossHistory', 'validationLossHistory', 'epochHistory', ...
    'epochs', 'bestEpoch', 'stoppedEarly', 'trainingOptions'});
save(weightsPath, 'network', 'inferenceConfig', 'trainingSummary', '-mat7-binary');
fprintf('Saved network weights and inference metadata to %s\n', weightsPath);

if options.exportEvaluationArtifacts && testMetrics.voxelIoU > 0
    exportHeldOutArtifacts(network, dataset, weightsDirectory, testProbabilities);
end
end

function options = setDefault(options, fieldName, defaultValue)
if ~isstruct(options) || ~isscalar(options)
    error('options must be a scalar structure.');
end
if ~isfield(options, fieldName) || isempty(options.(fieldName))
    options.(fieldName) = defaultValue;
end
end

function validateOptions(options)
positiveIntegers = {'hiddenUnits', 'maxEpochs', 'batchSize', 'patience', 'seed'};
for fieldIndex = 1:numel(positiveIntegers)
    fieldName = positiveIntegers{fieldIndex};
    value = options.(fieldName);
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value < 1 || value ~= floor(value)
        error('options.%s must be a finite positive integer.', fieldName);
    end
end
positiveValues = {'learningRate', 'maxGradientNorm', 'positiveClassWeight'};
for fieldIndex = 1:numel(positiveValues)
    fieldName = positiveValues{fieldIndex};
    value = options.(fieldName);
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value <= 0
        error('options.%s must be a finite positive scalar.', fieldName);
    end
end
nonnegativeValues = {'minImprovement', 'weightDecay'};
for fieldIndex = 1:numel(nonnegativeValues)
    fieldName = nonnegativeValues{fieldIndex};
    value = options.(fieldName);
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value < 0
        error('options.%s must be a finite nonnegative scalar.', fieldName);
    end
end
if ~islogical(options.exportEvaluationArtifacts) || ...
        ~isscalar(options.exportEvaluationArtifacts)
    error('options.exportEvaluationArtifacts must be a logical scalar.');
end
end

function validateDatasetSplits(dataset, config)
splitNames = {'training', 'validation', 'test'};
for splitIndex = 1:numel(splitNames)
    splitName = splitNames{splitIndex};
    inputs = dataset.([splitName, 'Inputs']);
    targets = dataset.([splitName, 'Targets']);
    if ~isnumeric(inputs) || ~isreal(inputs) || ndims(inputs) ~= 2 || ...
            size(inputs, 2) ~= config.inputSize ^ 2 || isempty(inputs) || ...
            any(~isfinite(inputs(:))) || any(inputs(:) < 0) || any(inputs(:) > 1)
        error('%s inputs do not match the dataset image size.', splitName);
    end
    if ~isnumeric(targets) || ~isreal(targets) || ndims(targets) ~= 2 || ...
            size(targets, 1) ~= size(inputs, 1) || ...
            size(targets, 2) ~= config.voxelResolution ^ 3 || ...
            any(~isfinite(targets(:))) || ...
            any((targets(:) ~= 0) & (targets(:) ~= 1))
        error('%s targets do not match the dataset voxel resolution.', splitName);
    end
end
end

function text = yesNo(value)
if value
    text = 'yes';
else
    text = 'no';
end
end