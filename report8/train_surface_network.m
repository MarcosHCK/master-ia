function network = train_surface_network(datasetFile, modelFile, learningRate, ...
        maxEpochs, patience, minImprovement, batchSize)
%TRAIN_SURFACE_NETWORK Train the 2-D surface CNN with early stopping.
%   TRAIN_SURFACE_NETWORK() reads examples.mat beside this file, trains with
%   Adam and validation RMSE early stopping, and writes trained-network.mat.
%   Optional arguments: datasetFile, modelFile, learningRate, maxEpochs,
%   patience, minImprovement, batchSize. The saved MAT-file contains network.
%   Requires only base MATLAB/Octave functions.

scriptDirectory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(datasetFile)
    datasetFile = fullfile(scriptDirectory, 'examples.mat');
end
if nargin < 2 || isempty(modelFile)
    modelFile = fullfile(scriptDirectory, 'trained-network.mat');
end
if nargin < 3 || isempty(learningRate)
    learningRate = 0.003;
end
if nargin < 4 || isempty(maxEpochs)
    maxEpochs = 200;
end
if nargin < 5 || isempty(patience)
    patience = 20;
end
if nargin < 6 || isempty(minImprovement)
    minImprovement = 1e-5;
end
if nargin < 7 || isempty(batchSize)
    batchSize = 16;
end
if ~ischar(datasetFile) || ~ischar(modelFile)
    error('datasetFile and modelFile must be character vectors.');
end
if ~isnumeric(learningRate) || ~isscalar(learningRate) || ...
        ~surface_isfinite(learningRate) || learningRate <= 0
    error('learningRate must be a finite positive scalar.');
end
if ~isnumeric(maxEpochs) || ~isscalar(maxEpochs) || ...
        ~surface_isfinite(maxEpochs) || ...
        maxEpochs < 1 || maxEpochs ~= floor(maxEpochs)
    error('maxEpochs must be a finite positive integer.');
end
if ~isnumeric(patience) || ~isscalar(patience) || ~surface_isfinite(patience) || ...
        patience < 1 || patience ~= floor(patience)
    error('patience must be a finite positive integer.');
end
if ~isnumeric(minImprovement) || ~isscalar(minImprovement) || ...
        ~surface_isfinite(minImprovement) || minImprovement < 0
    error('minImprovement must be a finite nonnegative scalar.');
end
if ~isnumeric(batchSize) || ~isscalar(batchSize) || ...
        ~surface_isfinite(batchSize) || ...
        batchSize < 1 || batchSize ~= floor(batchSize)
    error('batchSize must be a finite positive integer.');
end
if exist(datasetFile, 'file') ~= 2
    error('Dataset not found: %s. Run generate_surface_dataset first.', datasetFile);
end
loaded = load(datasetFile, 'dataset');
if ~isfield(loaded, 'dataset')
    error('Dataset file must contain a variable named dataset.');
end
dataset = loaded.dataset;
validateSplit(dataset, 'train');
validateSplit(dataset, 'validation');
hasTestSplit = isfield(dataset, 'test') && ...
    isfield(dataset.test, 'inputImages') && isfield(dataset.test, 'targetHeights');
if hasTestSplit
    validateSplit(dataset, 'test');
end
N = dataset.N;

trainImages = reshape(double(dataset.train.inputImages), N, N, 1, []);
trainTargets = reshape(double(dataset.train.targetHeights) + 0.5, N, N, 1, []);
validationImages = reshape(double(dataset.validation.inputImages), N, N, 1, []);
validationTargets = reshape(double(dataset.validation.targetHeights) + 0.5, ...
    N, N, 1, []);
if any(trainImages(:) < 0) || any(trainImages(:) > 1) || ...
        any(validationImages(:) < 0) || any(validationImages(:) > 1) || ...
        any(trainTargets(:) < 0) || any(trainTargets(:) > 1) || ...
        any(validationTargets(:) < 0) || any(validationTargets(:) > 1)
    error('Dataset images and target heights must map to intensities in [0,1].');
end

seed = 1;
if isfield(dataset, 'seed')
    seed = dataset.seed + 1;
end
network = surface_cnn_init(N, seed);
firstMoment = initializeOptimizerState(network);
secondMoment = initializeOptimizerState(network);
previousRngState = rng;
rng(seed + 2, 'twister');
rngCleanup = onCleanup(@() rng(previousRngState));
beta1 = 0.9;
beta2 = 0.999;
adamEpsilon = 1e-8;
optimizerStep = 0;
numTrainingExamples = size(trainImages, 4);
bestValidationRMSE = Inf;
epochsWithoutImprovement = 0;
trainingRMSEHistory = zeros(maxEpochs, 1);
validationRMSEHistory = zeros(maxEpochs, 1);
stoppedEarly = false;

for epoch = 1:maxEpochs
    shuffledExamples = randperm(numTrainingExamples);
    for firstExample = 1:batchSize:numTrainingExamples
        batchExamples = shuffledExamples(firstExample:min( ...
            firstExample + batchSize - 1, numTrainingExamples));
        batchImages = trainImages(:, :, :, batchExamples);
        batchTargets = trainTargets(:, :, :, batchExamples);
        [trainingPrediction, cache] = surface_cnn_forward(network, batchImages);
        trainingError = trainingPrediction - batchTargets;
        gradients = calculateGradients(network, cache, trainingError);
        optimizerStep = optimizerStep + 1;
        [network, firstMoment, secondMoment] = adamUpdate(network, gradients, ...
            firstMoment, secondMoment, optimizerStep, learningRate, beta1, ...
            beta2, adamEpsilon);
    end

    trainingRMSE = evaluateDataset(network, trainImages, trainTargets, batchSize);
    validationRMSE = evaluateDataset(network, validationImages, ...
        validationTargets, batchSize);
    trainingRMSEHistory(epoch) = trainingRMSE;
    validationRMSEHistory(epoch) = validationRMSE;

    % Save this checkpoint; restore the best validation model after training.
    if validationRMSE < bestValidationRMSE - minImprovement
        bestValidationRMSE = validationRMSE;
        bestEpoch = epoch;
        bestNetwork = network;
        bestTrainingRMSE = trainingRMSE;
        epochsWithoutImprovement = 0;
    else
        epochsWithoutImprovement = epochsWithoutImprovement + 1;
        if epochsWithoutImprovement >= patience
            stoppedEarly = true;
            break;
        end
    end

    if mod(epoch, 10) == 0 || epoch == 1
        fprintf('Epoch %d/%d: train RMSE %.6f, validation RMSE %.6f\n', ...
            epoch, maxEpochs, trainingRMSE, validationRMSE);
    end
end

epochsRun = epoch;
network = bestNetwork;
network.trainingRMSE = bestTrainingRMSE;
network.validationRMSE = bestValidationRMSE;
network.trainingRMSEHistory = trainingRMSEHistory(1:epochsRun);
network.validationRMSEHistory = validationRMSEHistory(1:epochsRun);
network.epochHistory = (1:epochsRun)';
network.epochs = epochsRun;
network.bestEpoch = bestEpoch;
network.stoppedEarly = stoppedEarly;
network.learningRate = learningRate;
network.patience = patience;
network.minImprovement = minImprovement;
network.batchSize = batchSize;
if hasTestSplit
    testImages = reshape(double(dataset.test.inputImages), N, N, 1, []);
    testTargets = reshape(double(dataset.test.targetHeights) + 0.5, N, N, 1, []);
    network.testRMSE = evaluateDataset(network, testImages, testTargets, batchSize);
else
    network.testRMSE = NaN;
end

modelDirectory = fileparts(modelFile);
if ~isempty(modelDirectory) && exist(modelDirectory, 'dir') ~= 7
    error('The model output directory does not exist: %s', modelDirectory);
end
save(modelFile, 'network', '-v7');
clear rngCleanup;
fprintf('Saved best model to %s (epoch %d, validation RMSE %.6f).\n', ...
    modelFile, bestEpoch, bestValidationRMSE);
if hasTestSplit
    fprintf('Held-out test RMSE: %.6f.\n', network.testRMSE);
end
if stoppedEarly
    fprintf('Early stopping after %d epochs (patience %d).\n', epochsRun, patience);
end
end

function rmse = evaluateDataset(network, images, targets, batchSize)
numExamples = size(images, 4);
sumSquaredError = 0;
numValues = 0;
for firstExample = 1:batchSize:numExamples
    lastExample = min(firstExample + batchSize - 1, numExamples);
    examples = firstExample:lastExample;
    prediction = surface_cnn_predict(network, ...
        reshape(images(:, :, :, examples), network.N, network.N, numel(examples)));
    errorValues = reshape(prediction, network.N, network.N, 1, []) - ...
        targets(:, :, :, examples);
    sumSquaredError = sumSquaredError + sum(errorValues(:).^2);
    numValues = numValues + numel(errorValues);
end
rmse = sqrt(sumSquaredError / numValues);
end

function validateSplit(dataset, splitName)
if ~isfield(dataset, 'N') || ~isfield(dataset, splitName) || ...
        ~isfield(dataset.(splitName), 'inputImages') || ...
        ~isfield(dataset.(splitName), 'targetHeights')
    error('Dataset must have N and %s inputImages/targetHeights.', splitName);
end
N = dataset.N;
images = dataset.(splitName).inputImages;
targets = dataset.(splitName).targetHeights;
if ~isnumeric(N) || ~isscalar(N) || N < 1 || ...
        size(images, 1) ~= N || size(images, 2) ~= N || ...
        size(targets, 1) ~= N || size(targets, 2) ~= N || ...
        size(images, 3) < 1 || size(targets, 3) ~= size(images, 3) || ...
        any(~surface_isfinite(images(:))) || any(~surface_isfinite(targets(:)))
    error('Dataset %s images and targets must be finite N-by-N-by-example arrays.', ...
        splitName);
end
end

function state = initializeOptimizerState(network)
state.stemWeights = zeros(size(network.stemWeights));
state.stemBias = zeros(size(network.stemBias));
state.blocks(1).conv1Weights = zeros(size(network.blocks(1).conv1Weights));
state.blocks(1).conv1Bias = zeros(size(network.blocks(1).conv1Bias));
state.blocks(1).conv2Weights = zeros(size(network.blocks(1).conv2Weights));
state.blocks(1).conv2Bias = zeros(size(network.blocks(1).conv2Bias));
state.headWeights = zeros(size(network.headWeights));
state.headBias = zeros(size(network.headBias));
end

function gradients = calculateGradients(network, cache, predictionError)
outputDerivative = double(cache.unclippedPrediction > 0 & ...
    cache.unclippedPrediction < 1);
headOutputGradient = (2 / numel(predictionError)) * predictionError .* ...
    outputDerivative * network.residualScale;
[blockOutputGradient, gradients.headWeights, gradients.headBias] = ...
    surface_cnn_conv2d_backward(cache.blockOutput, network.headWeights, ...
    headOutputGradient, 1);

blockResidualGradient = blockOutputGradient .* (cache.blockResidualSum > 0);
[firstActivationGradient, gradients.blocks(1).conv2Weights, ...
    gradients.blocks(1).conv2Bias] = surface_cnn_conv2d_backward( ...
    cache.blockFirstActivation, network.blocks(1).conv2Weights, ...
    blockResidualGradient, 1);
firstLinearGradient = firstActivationGradient .* (cache.blockFirstLinear > 0);
[blockInputGradient, gradients.blocks(1).conv1Weights, ...
    gradients.blocks(1).conv1Bias] = surface_cnn_conv2d_backward( ...
    cache.blockInput, network.blocks(1).conv1Weights, firstLinearGradient, 1);
stemOutputGradient = (blockResidualGradient + blockInputGradient) .* ...
    (cache.stemLinear > 0);
[~, gradients.stemWeights, gradients.stemBias] = ...
    surface_cnn_conv2d_backward(cache.input, network.stemWeights, ...
    stemOutputGradient, 1);
end

function [network, firstMoment, secondMoment] = adamUpdate(network, gradients, ...
        firstMoment, secondMoment, step, learningRate, beta1, beta2, epsilon)
parameterNames = {'stemWeights', 'stemBias', 'headWeights', 'headBias'};
for index = 1:numel(parameterNames)
    name = parameterNames{index};
    [network.(name), firstMoment.(name), secondMoment.(name)] = ...
        updateParameter(network.(name), gradients.(name), firstMoment.(name), ...
        secondMoment.(name), step, learningRate, beta1, beta2, epsilon);
end
blockNames = {'conv1Weights', 'conv1Bias', 'conv2Weights', 'conv2Bias'};
for index = 1:numel(blockNames)
    name = blockNames{index};
    [network.blocks(1).(name), firstMoment.blocks(1).(name), ...
        secondMoment.blocks(1).(name)] = updateParameter( ...
        network.blocks(1).(name), gradients.blocks(1).(name), ...
        firstMoment.blocks(1).(name), secondMoment.blocks(1).(name), ...
        step, learningRate, beta1, beta2, epsilon);
end
end

function [parameter, first, second] = updateParameter(parameter, gradient, ...
        first, second, step, learningRate, beta1, beta2, epsilon)
first = beta1 * first + (1 - beta1) * gradient;
second = beta2 * second + (1 - beta2) * (gradient.^2);
correctedFirst = first / (1 - beta1^step);
correctedSecond = second / (1 - beta2^step);
parameter = parameter - learningRate * correctedFirst ./ ...
    (sqrt(correctedSecond) + epsilon);
end