function network = trainReconstructionNetwork(network, wrappedTrainingInputs, originalTrainingTargets, learningRate, maxEpochs, wrappedValidationInputs, originalValidationTargets, patience, minImprovement)
%TRAINRECONSTRUCTIONNETWORK Fit the network with validation-based early stopping.
% Every training row contains one N-sample wrapped signal or target signal.

if nargin < 4
    learningRate = 0.2;
end
if nargin < 5
    maxEpochs = 3000;
end
if nargin < 6 || isempty(wrappedValidationInputs)
    wrappedValidationInputs = wrappedTrainingInputs;
    originalValidationTargets = originalTrainingTargets;
elseif nargin < 7 || isempty(originalValidationTargets)
    error('Both validation inputs and validation targets must be provided.');
end
if nargin < 8 || isempty(patience)
    patience = 500;
end
if nargin < 9 || isempty(minImprovement)
    minImprovement = 1e-6;
end
if ~isstruct(network) || ~isfield(network, 'N') || ...
        ~isfield(network, 'hiddenUnits') || ...
        ~isfield(network, 'inputHiddenWeights') || ...
        ~isfield(network, 'hiddenBias') || ...
        ~isfield(network, 'hiddenOutputWeights') || ...
        ~isfield(network, 'outputBias')
    error('network must be a model returned by buildNetwork.');
end
N = network.N;
hiddenUnits = network.hiddenUnits;
if ~isnumeric(wrappedTrainingInputs) || ~isreal(wrappedTrainingInputs) || ...
        ndims(wrappedTrainingInputs) ~= 2 || size(wrappedTrainingInputs, 2) ~= N || ...
        any(isnan(wrappedTrainingInputs(:))) || any(isinf(wrappedTrainingInputs(:)))
    error('wrappedTrainingInputs must be a finite matrix with N columns.');
end
if ~isnumeric(originalTrainingTargets) || ~isreal(originalTrainingTargets) || ...
        ~isequal(size(originalTrainingTargets), size(wrappedTrainingInputs)) || ...
        any(isnan(originalTrainingTargets(:))) || any(isinf(originalTrainingTargets(:)))
    error('originalTrainingTargets must be a finite matrix matching the input size.');
end
if size(wrappedTrainingInputs, 1) < 1
    error('At least one training example is required.');
end
if ~isnumeric(wrappedValidationInputs) || ~isreal(wrappedValidationInputs) || ...
        ndims(wrappedValidationInputs) ~= 2 || ...
        size(wrappedValidationInputs, 2) ~= N || ...
        any(isnan(wrappedValidationInputs(:))) || ...
        any(isinf(wrappedValidationInputs(:)))
    error('wrappedValidationInputs must be a finite matrix with N columns.');
end
if ~isnumeric(originalValidationTargets) || ~isreal(originalValidationTargets) || ...
        ~isequal(size(originalValidationTargets), size(wrappedValidationInputs)) || ...
        any(isnan(originalValidationTargets(:))) || ...
        any(isinf(originalValidationTargets(:)))
    error('originalValidationTargets must be a finite matrix matching the validation inputs.');
end
if size(wrappedValidationInputs, 1) < 1
    error('At least one validation example is required.');
end
if ~isnumeric(learningRate) || ~isreal(learningRate) || ...
        ~isscalar(learningRate) || isnan(learningRate) || ...
        isinf(learningRate) || learningRate <= 0
    error('learningRate must be a finite, positive scalar.');
end
if ~isnumeric(maxEpochs) || ~isreal(maxEpochs) || ...
        ~isscalar(maxEpochs) || isnan(maxEpochs) || ...
        isinf(maxEpochs) || maxEpochs < 1 || maxEpochs ~= floor(maxEpochs)
    error('maxEpochs must be a finite positive integer.');
end
if ~isnumeric(patience) || ~isreal(patience) || ...
        ~isscalar(patience) || isnan(patience) || isinf(patience) || ...
        patience < 1 || patience ~= floor(patience)
    error('patience must be a finite positive integer.');
end
if ~isnumeric(minImprovement) || ~isreal(minImprovement) || ...
        ~isscalar(minImprovement) || isnan(minImprovement) || ...
        isinf(minImprovement) || minImprovement < 0
    error('minImprovement must be a finite, non-negative scalar.');
end
if ~isequal(size(network.inputHiddenWeights), [N, hiddenUnits]) || ...
        numel(network.hiddenBias) ~= hiddenUnits || ...
        ~isequal(size(network.hiddenOutputWeights), [hiddenUnits, N]) || ...
        numel(network.outputBias) ~= N
    error('The network parameters do not match its input, hidden, and output sizes.');
end

trainingExampleCount = size(wrappedTrainingInputs, 1);
validationExampleCount = size(wrappedValidationInputs, 1);
trainingRMSEHistory = zeros(maxEpochs, 1);
bestValidationRMSE = Inf;
epochsWithoutImprovement = 0;
stoppedEarly = false;
for epoch = 1:maxEpochs
    hiddenLinear = wrappedTrainingInputs * network.inputHiddenWeights + ...
        repmat(reshape(network.hiddenBias, 1, hiddenUnits), trainingExampleCount, 1);
    hiddenActivations = tanh(hiddenLinear);
    outputLinear = hiddenActivations * network.hiddenOutputWeights + ...
        repmat(reshape(network.outputBias, 1, N), trainingExampleCount, 1);
    predictedTargets = 2 * min(1, max(0, outputLinear)) - 1;
    outputError = predictedTargets - originalTrainingTargets;
    trainingRMSEHistory(epoch) = sqrt(mean(outputError(:).^2));

    % Backpropagate MSE through the capped output ReLU and hidden tanh.
    outputGradient = (4 / numel(outputError)) * outputError .* ...
        (outputLinear > 0 & outputLinear < 1);
    hiddenGradient = (outputGradient * network.hiddenOutputWeights') .* ...
        (1 - hiddenActivations.^2);
    inputHiddenWeightGradient = wrappedTrainingInputs' * hiddenGradient;
    hiddenBiasGradient = sum(hiddenGradient, 1);
    hiddenOutputWeightGradient = hiddenActivations' * outputGradient;
    outputBiasGradient = sum(outputGradient, 1);

    network.inputHiddenWeights = network.inputHiddenWeights - ...
        learningRate * inputHiddenWeightGradient;
    network.hiddenBias = network.hiddenBias - learningRate * hiddenBiasGradient;
    network.hiddenOutputWeights = network.hiddenOutputWeights - ...
        learningRate * hiddenOutputWeightGradient;
    network.outputBias = network.outputBias - learningRate * outputBiasGradient;

    validationHiddenLinear = wrappedValidationInputs * network.inputHiddenWeights + ...
        repmat(reshape(network.hiddenBias, 1, hiddenUnits), validationExampleCount, 1);
    validationHiddenActivations = tanh(validationHiddenLinear);
    validationOutputLinear = validationHiddenActivations * ...
        network.hiddenOutputWeights + ...
        repmat(reshape(network.outputBias, 1, N), validationExampleCount, 1);
    validationPrediction = 2 * min(1, max(0, validationOutputLinear)) - 1;
    currentValidationRMSE = sqrt(mean((validationPrediction(:) - ...
        originalValidationTargets(:)).^2));

    if currentValidationRMSE < bestValidationRMSE - minImprovement
        bestValidationRMSE = currentValidationRMSE;
        bestEpoch = epoch;
        bestInputHiddenWeights = network.inputHiddenWeights;
        bestHiddenBias = network.hiddenBias;
        bestHiddenOutputWeights = network.hiddenOutputWeights;
        bestOutputBias = network.outputBias;
        epochsWithoutImprovement = 0;
    else
        epochsWithoutImprovement = epochsWithoutImprovement + 1;
        if epochsWithoutImprovement >= patience
            stoppedEarly = true;
            break;
        end
    end
end

epochsRun = epoch;
network.inputHiddenWeights = bestInputHiddenWeights;
network.hiddenBias = bestHiddenBias;
network.hiddenOutputWeights = bestHiddenOutputWeights;
network.outputBias = bestOutputBias;

finalHiddenLinear = wrappedTrainingInputs * network.inputHiddenWeights + ...
    repmat(reshape(network.hiddenBias, 1, hiddenUnits), trainingExampleCount, 1);
finalHiddenActivations = tanh(finalHiddenLinear);
finalOutputLinear = finalHiddenActivations * network.hiddenOutputWeights + ...
    repmat(reshape(network.outputBias, 1, N), trainingExampleCount, 1);
finalPrediction = 2 * min(1, max(0, finalOutputLinear)) - 1;
network.trainingRMSE = sqrt(mean((finalPrediction(:) - ...
    originalTrainingTargets(:)).^2));
network.trainingRMSEHistory = trainingRMSEHistory(1:epochsRun);
network.epochHistory = (1:epochsRun)';
network.validationRMSE = bestValidationRMSE;
network.epochs = epochsRun;
network.bestEpoch = bestEpoch;
network.stoppedEarly = stoppedEarly;
end