function network = trainCNNNetwork(network, wrappedTrainingInputs, originalTrainingTargets, learningRate, maxEpochs, wrappedValidationInputs, originalValidationTargets, patience, minImprovement)
%TRAINCNNNETWORK Train the residual CNN using validation early stopping.

if nargin < 6 || isempty(wrappedValidationInputs) || nargin < 7 || isempty(originalValidationTargets)
    error('CNN training requires explicit validation inputs and targets.');
end
if nargin < 8 || isempty(patience)
    patience = 500;
end
if nargin < 9 || isempty(minImprovement)
    minImprovement = 1e-6;
end
if ~isnumeric(wrappedTrainingInputs) || ~isreal(wrappedTrainingInputs) || ...
        ndims(wrappedTrainingInputs) ~= 2 || size(wrappedTrainingInputs, 2) ~= network.N || ...
        any(~isfinite(wrappedTrainingInputs(:)))
    error('wrappedTrainingInputs must be a finite matrix with network.N columns.');
end
if ~isnumeric(originalTrainingTargets) || ~isreal(originalTrainingTargets) || ...
        ~isequal(size(originalTrainingTargets), size(wrappedTrainingInputs)) || ...
        any(~isfinite(originalTrainingTargets(:)))
    error('Training targets must be finite and match the training inputs.');
end
if ~isequal(size(wrappedValidationInputs), size(originalValidationTargets)) || ...
        size(wrappedValidationInputs, 2) ~= network.N || ...
        any(~isfinite(wrappedValidationInputs(:))) || ...
        any(~isfinite(originalValidationTargets(:)))
    error('Validation inputs and targets must be finite matrices with N columns.');
end
if ~isscalar(learningRate) || ~isfinite(learningRate) || learningRate <= 0 || ...
        ~isscalar(maxEpochs) || ~isfinite(maxEpochs) || maxEpochs < 1 || ...
        maxEpochs ~= floor(maxEpochs) || ~isscalar(patience) || ...
        ~isfinite(patience) || patience < 1 || patience ~= floor(patience) || ...
        ~isscalar(minImprovement) || ~isfinite(minImprovement) || minImprovement < 0
    error('Invalid CNN learning rate, epoch limit, patience, or minImprovement.');
end

trainingRMSEHistory = zeros(maxEpochs, 1);
validationRMSEHistory = zeros(maxEpochs, 1);
bestValidationRMSE = Inf;
epochsWithoutImprovement = 0;
stoppedEarly = false;

for epoch = 1:maxEpochs
    [prediction, cache] = cnnForward(network, wrappedTrainingInputs);
    errorValues = prediction - originalTrainingTargets;
    trainingRMSEHistory(epoch) = sqrt(mean(errorValues(:).^2));

    % Evaluate validation at the same pre-update checkpoint as training RMSE.
    [validationPrediction, ~] = cnnForward(network, wrappedValidationInputs);
    currentValidationRMSE = sqrt(mean((validationPrediction(:) - ...
        originalValidationTargets(:)).^2));
    validationRMSEHistory(epoch) = currentValidationRMSE;

    if currentValidationRMSE < bestValidationRMSE - minImprovement
        bestValidationRMSE = currentValidationRMSE;
        bestEpoch = epoch;
        bestNetwork = network;
        epochsWithoutImprovement = 0;
    else
        epochsWithoutImprovement = epochsWithoutImprovement + 1;
        if epochsWithoutImprovement >= patience
            stoppedEarly = true;
            break;
        end
    end

    outputGradient = (4 / numel(errorValues)) * errorValues .* ...
        (cache.outputLinear(:, :, 1) > 0 & cache.outputLinear(:, :, 1) < 1);
    outputGradient3d = reshape(outputGradient, size(outputGradient, 1), network.N, 1);

    [featureGradient, outputWeightGradient, outputBiasGradient] = ...
        cnnConv1dBackward(cache.finalFeatures, network.outputWeights, ...
        outputGradient3d, 1);

    blockGradients = cell(1, numel(network.dilations));
    for block = numel(network.dilations):-1:1
        blockCache = cache.blocks{block};
        sumGradient = featureGradient .* (blockCache.residualSum > 0);
        secondLinearGradient = sumGradient;
        [firstActivationGradient, conv2WeightGradient, conv2BiasGradient] = ...
            cnnConv1dBackward(blockCache.firstActivation, ...
            network.blocks(block).conv2Weights, secondLinearGradient, ...
            network.dilations(block));
        firstLinearGradient = firstActivationGradient .* (blockCache.firstLinear > 0);
        [residualGradient, conv1WeightGradient, conv1BiasGradient] = ...
            cnnConv1dBackward(blockCache.residualInput, ...
            network.blocks(block).conv1Weights, firstLinearGradient, ...
            network.dilations(block));
        featureGradient = sumGradient + residualGradient;
        blockGradients{block}.conv1Weights = conv1WeightGradient;
        blockGradients{block}.conv1Bias = conv1BiasGradient;
        blockGradients{block}.conv2Weights = conv2WeightGradient;
        blockGradients{block}.conv2Bias = conv2BiasGradient;
    end

    stemLinearGradient = featureGradient .* (cache.stemLinear > 0);
    [~, stemWeightGradient, stemBiasGradient] = cnnConv1dBackward( ...
        cache.input, network.stemWeights, stemLinearGradient, 1);

    % Clip the aggregate gradient to stabilize updates at larger learning rates.
    gradientNormSquared = sum(outputWeightGradient(:).^2) + ...
        sum(outputBiasGradient(:).^2) + ...
        sum(stemWeightGradient(:).^2) + sum(stemBiasGradient(:).^2);
    for block = 1:numel(network.dilations)
        gradientNormSquared = gradientNormSquared + ...
            sum(blockGradients{block}.conv1Weights(:).^2) + ...
            sum(blockGradients{block}.conv1Bias(:).^2) + ...
            sum(blockGradients{block}.conv2Weights(:).^2) + ...
            sum(blockGradients{block}.conv2Bias(:).^2);
    end
    gradientScale = min(1, network.maxGradientNorm / ...
        max(sqrt(gradientNormSquared), eps));
    outputWeightGradient = gradientScale * outputWeightGradient;
    outputBiasGradient = gradientScale * outputBiasGradient;
    stemWeightGradient = gradientScale * stemWeightGradient;
    stemBiasGradient = gradientScale * stemBiasGradient;
    for block = 1:numel(network.dilations)
        blockGradients{block}.conv1Weights = ...
            gradientScale * blockGradients{block}.conv1Weights;
        blockGradients{block}.conv1Bias = ...
            gradientScale * blockGradients{block}.conv1Bias;
        blockGradients{block}.conv2Weights = ...
            gradientScale * blockGradients{block}.conv2Weights;
        blockGradients{block}.conv2Bias = ...
            gradientScale * blockGradients{block}.conv2Bias;
    end

    network.outputWeights = network.outputWeights - learningRate * outputWeightGradient;
    network.outputBias = network.outputBias - learningRate * outputBiasGradient;
    network.stemWeights = network.stemWeights - learningRate * stemWeightGradient;
    network.stemBias = network.stemBias - learningRate * stemBiasGradient;
    for block = 1:numel(network.dilations)
        network.blocks(block).conv1Weights = network.blocks(block).conv1Weights - ...
            learningRate * blockGradients{block}.conv1Weights;
        network.blocks(block).conv1Bias = network.blocks(block).conv1Bias - ...
            learningRate * blockGradients{block}.conv1Bias;
        network.blocks(block).conv2Weights = network.blocks(block).conv2Weights - ...
            learningRate * blockGradients{block}.conv2Weights;
        network.blocks(block).conv2Bias = network.blocks(block).conv2Bias - ...
            learningRate * blockGradients{block}.conv2Bias;
    end

    if mod(epoch, 250) == 0
        fprintf('CNN epoch %d: training RMSE %.6f, validation RMSE %.6f\n', ...
            epoch, trainingRMSEHistory(epoch), validationRMSEHistory(epoch));
        fflush(stdout);
    end
end

epochsRun = epoch;
network = bestNetwork;
network.trainingRMSE = trainingRMSEHistory(bestEpoch);
network.validationRMSE = bestValidationRMSE;
network.trainingRMSEHistory = trainingRMSEHistory(1:epochsRun);
network.validationRMSEHistory = validationRMSEHistory(1:epochsRun);
network.epochHistory = (1:epochsRun)';
network.epochs = epochsRun;
network.bestEpoch = bestEpoch;
network.stoppedEarly = stoppedEarly;
end

function [prediction, cache] = cnnForward(network, wrappedInputs)
batchSize = size(wrappedInputs, 1);
cache.input = reshape(wrappedInputs, batchSize, network.N, 1);
cache.stemLinear = cnnConv1d(cache.input, network.stemWeights, network.stemBias, 1);
features = max(0, cache.stemLinear);
cache.blocks = cell(1, numel(network.dilations));
for block = 1:numel(network.dilations)
    cache.blocks{block}.residualInput = features;
    cache.blocks{block}.firstLinear = cnnConv1d(features, ...
        network.blocks(block).conv1Weights, network.blocks(block).conv1Bias, ...
        network.dilations(block));
    cache.blocks{block}.firstActivation = max(0, cache.blocks{block}.firstLinear);
    cache.blocks{block}.secondLinear = cnnConv1d(cache.blocks{block}.firstActivation, ...
        network.blocks(block).conv2Weights, network.blocks(block).conv2Bias, ...
        network.dilations(block));
    cache.blocks{block}.residualSum = features + cache.blocks{block}.secondLinear;
    features = max(0, cache.blocks{block}.residualSum);
end
cache.finalFeatures = features;
cache.outputLinear = cnnConv1d(features, network.outputWeights, network.outputBias, 1);
prediction = 2 * min(1, max(0, cache.outputLinear(:, :, 1))) - 1;
end