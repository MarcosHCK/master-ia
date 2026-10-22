function network = trainNetwork(network, trainingInputs, trainingTargets, ...
        validationInputs, validationTargets, options)
%TRAINNETWORK Train the image-to-height-map MLP with mini-batch backprop.

if nargin < 6 || isempty(options)
    options = struct();
end
options = setDefault(options, 'learningRate', 0.01);
options = setDefault(options, 'maxEpochs', 300);
options = setDefault(options, 'batchSize', 32);
options = setDefault(options, 'patience', 30);
options = setDefault(options, 'minImprovement', 1e-5);
options = setDefault(options, 'weightDecay', 1e-5);
options = setDefault(options, 'maxGradientNorm', 10);
options = setDefault(options, 'seed', 1);

validateNetwork(network);
inputFeatures = network.inputSize ^ 2;
outputFeatures = network.surfaceResolution ^ 2;
validateData(trainingInputs, trainingTargets, inputFeatures, outputFeatures, 'training');
validateData(validationInputs, validationTargets, inputFeatures, outputFeatures, 'validation');
if ~isstruct(options) || ~isscalar(options) || ...
        ~isscalar(options.learningRate) || ~isfinite(options.learningRate) || ...
        options.learningRate <= 0 || ~isscalar(options.maxEpochs) || ...
        ~isfinite(options.maxEpochs) || options.maxEpochs < 1 || ...
        options.maxEpochs ~= floor(options.maxEpochs) || ...
        ~isscalar(options.batchSize) || options.batchSize < 1 || ...
        options.batchSize ~= floor(options.batchSize) || ...
        ~isscalar(options.patience) || options.patience < 1 || ...
        options.patience ~= floor(options.patience) || ...
        ~isscalar(options.minImprovement) || ~isfinite(options.minImprovement) || ...
        options.minImprovement < 0 || ~isscalar(options.weightDecay) || ...
        ~isfinite(options.weightDecay) || options.weightDecay < 0 || ...
        ~isscalar(options.maxGradientNorm) || ~isfinite(options.maxGradientNorm) || ...
        options.maxGradientNorm <= 0 || ~isscalar(options.seed) || ...
        ~isfinite(options.seed) || options.seed ~= floor(options.seed)
    error('Invalid training options.');
end

previousRngState = rng;
rng(options.seed, 'twister');
restoreRng = onCleanup(@() rng(previousRngState));

trainingCount = size(trainingInputs, 1);
batchSize = min(options.batchSize, trainingCount);
trainingLossHistory = zeros(options.maxEpochs, 1);
validationLossHistory = zeros(options.maxEpochs, 1);
bestValidationLoss = Inf;
epochsWithoutImprovement = 0;
stoppedEarly = false;
nextProgressPercent = 5;

for epoch = 1:options.maxEpochs
    shuffledIndices = randperm(trainingCount);
    for firstIndex = 1:batchSize:trainingCount
        batchIndices = shuffledIndices(firstIndex:min(firstIndex + batchSize - 1, trainingCount));
        batchInputs = trainingInputs(batchIndices, :);
        batchTargets = trainingTargets(batchIndices, :);

        centeredInputs = batchInputs - network.inputMean;
        hiddenLinear = centeredInputs * network.W1 + ...
            repmat(network.b1, numel(batchIndices), 1);
        hiddenActivations = max(0, hiddenLinear);
        outputLogits = hiddenActivations * network.W2 + ...
            repmat(network.b2, numel(batchIndices), 1);
        predictions = stableSigmoid(outputLogits);

        % Mean squared height error averages over examples and surface points.
        outputGradient = (2 / numel(batchTargets)) * ...
            (predictions - batchTargets) .* predictions .* (1 - predictions);
        weight2Gradient = hiddenActivations' * outputGradient + ...
            options.weightDecay * network.W2;
        bias2Gradient = sum(outputGradient, 1);
        hiddenGradient = (outputGradient * network.W2') .* (hiddenLinear > 0);
        weight1Gradient = centeredInputs' * hiddenGradient + ...
            options.weightDecay * network.W1;
        bias1Gradient = sum(hiddenGradient, 1);

        gradientNorm = sqrt(sum(weight1Gradient(:) .^ 2) + ...
            sum(bias1Gradient(:) .^ 2) + sum(weight2Gradient(:) .^ 2) + ...
            sum(bias2Gradient(:) .^ 2));
        gradientScale = min(1, options.maxGradientNorm / max(gradientNorm, realmin));
        network.W1 = network.W1 - options.learningRate * gradientScale * weight1Gradient;
        network.b1 = network.b1 - options.learningRate * gradientScale * bias1Gradient;
        network.W2 = network.W2 - options.learningRate * gradientScale * weight2Gradient;
        network.b2 = network.b2 - options.learningRate * gradientScale * bias2Gradient;
    end

    trainingLossHistory(epoch) = meanSquaredError(network, trainingInputs, trainingTargets);
    currentValidationLoss = meanSquaredError(network, validationInputs, validationTargets);
    validationLossHistory(epoch) = currentValidationLoss;

    progressPercent = floor(100 * epoch / options.maxEpochs);
    while nextProgressPercent <= 100 && progressPercent >= nextProgressPercent
        fprintf('Training progress: %d%% (epoch %d/%d; train MSE %.6f, validation MSE %.6f)\n', ...
            nextProgressPercent, epoch, options.maxEpochs, ...
            trainingLossHistory(epoch), currentValidationLoss);
        fflush(stdout);
        nextProgressPercent = nextProgressPercent + 5;
    end

    if currentValidationLoss < bestValidationLoss - options.minImprovement
        bestValidationLoss = currentValidationLoss;
        bestEpoch = epoch;
        bestNetwork = network;
        epochsWithoutImprovement = 0;
    else
        epochsWithoutImprovement = epochsWithoutImprovement + 1;
        if epochsWithoutImprovement >= options.patience
            stoppedEarly = true;
            break;
        end
    end
end

epochsRun = epoch;
network = bestNetwork;
network.trainingLoss = meanSquaredError(network, trainingInputs, trainingTargets);
network.validationLoss = bestValidationLoss;
network.trainingLossHistory = trainingLossHistory(1:epochsRun);
network.validationLossHistory = validationLossHistory(1:epochsRun);
network.epochHistory = (1:epochsRun)';
network.epochs = epochsRun;
network.bestEpoch = bestEpoch;
network.stoppedEarly = stoppedEarly;
network.trainingOptions = options;

clear restoreRng;
end

function options = setDefault(options, fieldName, defaultValue)
if ~isstruct(options) || ~isscalar(options)
    error('options must be a scalar structure.');
end
if ~isfield(options, fieldName) || isempty(options.(fieldName))
    options.(fieldName) = defaultValue;
end
end

function validateNetwork(network)
requiredFields = {'inputSize', 'surfaceResolution', 'hiddenUnits', ...
    'inputMean', 'W1', 'b1', 'W2', 'b2'};
if ~isstruct(network) || ~isscalar(network) || ...
        ~all(isfield(network, requiredFields))
    error('network must be a model returned by buildNetwork.');
end
if ~isequal(size(network.W1), [network.inputSize ^ 2, network.hiddenUnits]) || ...
        numel(network.b1) ~= network.hiddenUnits || ...
        ~isequal(size(network.W2), [network.hiddenUnits, network.surfaceResolution ^ 2]) || ...
        numel(network.b2) ~= network.surfaceResolution ^ 2
    error('Network parameter dimensions do not match the configured layer sizes.');
end
end

function validateData(inputs, targets, inputFeatures, outputFeatures, splitName)
if ~isnumeric(inputs) || ~isreal(inputs) || ndims(inputs) ~= 2 || ...
        size(inputs, 2) ~= inputFeatures || isempty(inputs) || ...
        any(~isfinite(inputs(:))) || any(inputs(:) < 0) || any(inputs(:) > 1)
    error('%s inputs must be finite grayscale depthmap rows in [0,1].', splitName);
end
if ~isnumeric(targets) || ~isreal(targets) || ndims(targets) ~= 2 || ...
        size(targets, 1) ~= size(inputs, 1) || ...
        size(targets, 2) ~= outputFeatures || any(~isfinite(targets(:))) || ...
        any(targets(:) < 0) || any(targets(:) > 1)
    error('%s targets must be matching normalized height-map rows in [0,1].', splitName);
end
end

function loss = meanSquaredError(network, inputs, targets)
predictions = applyNetwork(network, inputs);
loss = mean((predictions(:) - targets(:)) .^ 2);
end

function probabilities = stableSigmoid(logits)
clippedLogits = min(50, max(-50, logits));
probabilities = 1 ./ (1 + exp(-clippedLogits));
end