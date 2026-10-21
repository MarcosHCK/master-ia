function network = trainNetwork(network, trainingInputs, trainingTargets, ...
        validationInputs, validationTargets, options)
%TRAINNETWORK Train the image-to-voxel MLP with manual mini-batch backprop.

if nargin < 6 || isempty(options)
    options = struct();
end
options = setDefault(options, 'learningRate', 0.001);
options = setDefault(options, 'maxEpochs', 80);
options = setDefault(options, 'batchSize', 12);
options = setDefault(options, 'patience', 15);
options = setDefault(options, 'minImprovement', 1e-4);
options = setDefault(options, 'weightDecay', 1e-5);
options = setDefault(options, 'maxGradientNorm', 10);
options = setDefault(options, 'seed', 1);
options = setDefault(options, 'positiveClassWeight', 6);

validateNetwork(network);
inputFeatures = network.inputSize ^ 2;
outputFeatures = network.voxelResolution ^ 3;
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
        ~isfinite(options.seed) || options.seed ~= floor(options.seed) || ...
        ~isscalar(options.positiveClassWeight) || ...
        ~isfinite(options.positiveClassWeight) || options.positiveClassWeight <= 0
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

        % Binary cross-entropy is summed over voxels and averaged over examples.
        classWeights = 1 + (options.positiveClassWeight - 1) * batchTargets;
        outputGradient = classWeights .* (predictions - batchTargets) / ...
            numel(batchIndices);
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
        if gradientNorm > options.maxGradientNorm
            gradientScale = options.maxGradientNorm / gradientNorm;
            weight1Gradient = gradientScale * weight1Gradient;
            bias1Gradient = gradientScale * bias1Gradient;
            weight2Gradient = gradientScale * weight2Gradient;
            bias2Gradient = gradientScale * bias2Gradient;
        end

        network.W1 = network.W1 - options.learningRate * weight1Gradient;
        network.b1 = network.b1 - options.learningRate * bias1Gradient;
        network.W2 = network.W2 - options.learningRate * weight2Gradient;
        network.b2 = network.b2 - options.learningRate * bias2Gradient;
    end

    trainingLossHistory(epoch) = binaryCrossEntropy(network, trainingInputs, ...
        trainingTargets, options.positiveClassWeight);
    currentValidationLoss = binaryCrossEntropy(network, validationInputs, ...
        validationTargets, options.positiveClassWeight);
    validationLossHistory(epoch) = currentValidationLoss;

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
network.trainingLoss = binaryCrossEntropy(network, trainingInputs, ...
    trainingTargets, options.positiveClassWeight);
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
requiredFields = {'inputSize', 'voxelResolution', 'hiddenUnits', ...
    'inputMean', 'W1', 'b1', 'W2', 'b2'};
if ~isstruct(network) || ~isscalar(network) || ...
        ~all(isfield(network, requiredFields))
    error('network must be a model returned by buildNetwork.');
end
if ~isequal(size(network.W1), [network.inputSize ^ 2, network.hiddenUnits]) || ...
        numel(network.b1) ~= network.hiddenUnits || ...
        ~isequal(size(network.W2), [network.hiddenUnits, network.voxelResolution ^ 3]) || ...
        numel(network.b2) ~= network.voxelResolution ^ 3
    error('Network parameter dimensions do not match the configured layer sizes.');
end
end

function validateData(inputs, targets, inputFeatures, outputFeatures, splitName)
if ~isnumeric(inputs) || ~isreal(inputs) || ndims(inputs) ~= 2 || ...
        size(inputs, 2) ~= inputFeatures || isempty(inputs) || ...
        any(~isfinite(inputs(:))) || any(inputs(:) < 0) || any(inputs(:) > 1)
    error('%s inputs must be finite grayscale rows in [0,1].', splitName);
end
if ~isnumeric(targets) || ~isreal(targets) || ndims(targets) ~= 2 || ...
        size(targets, 1) ~= size(inputs, 1) || ...
        size(targets, 2) ~= outputFeatures || any(~isfinite(targets(:))) || ...
        any((targets(:) ~= 0) & (targets(:) ~= 1))
    error('%s targets must be matching binary voxel rows.', splitName);
end
end

function loss = binaryCrossEntropy(network, inputs, targets, positiveClassWeight)
centeredInputs = inputs - network.inputMean;
hiddenActivations = max(0, centeredInputs * network.W1 + ...
    repmat(network.b1, size(inputs, 1), 1));
logits = hiddenActivations * network.W2 + ...
    repmat(network.b2, size(inputs, 1), 1);
positiveLoss = max(-logits, 0) + log1p(exp(-abs(logits)));
negativeLoss = max(logits, 0) + log1p(exp(-abs(logits)));
perVoxelLoss = positiveClassWeight * targets .* positiveLoss + ...
    (1 - targets) .* negativeLoss;
loss = mean(sum(perVoxelLoss, 2));
end

function probabilities = stableSigmoid(logits)
clippedLogits = min(50, max(-50, logits));
probabilities = 1 ./ (1 + exp(-clippedLogits));
end