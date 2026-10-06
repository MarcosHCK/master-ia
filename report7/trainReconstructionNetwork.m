function network = trainReconstructionNetwork(network, wrappedTrainingInputs, originalTrainingTargets, learningRate, maxEpochs)
%TRAINRECONSTRUCTIONNETWORK Fit a sine-reconstruction network's weights.
% Every training row contains one N-sample wrapped signal or target signal.

if nargin < 4
    learningRate = 0.2;
end
if nargin < 5
    maxEpochs = 3000;
end
if ~isstruct(network) || ~isfield(network, 'N') || ...
        ~isfield(network, 'weights') || ~isfield(network, 'bias')
    error('network must be a model returned by buildNetwork.');
end
N = network.N;
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
if ~isequal(size(network.weights), [N, N]) || numel(network.bias) ~= N
    error('The network weights or bias do not match network.N.');
end

trainingExampleCount = size(wrappedTrainingInputs, 1);
for epoch = 1:maxEpochs
    linearOutput = wrappedTrainingInputs * network.weights + ...
        repmat(reshape(network.bias, 1, N), trainingExampleCount, 1);
    reluOutput = max(0, linearOutput);
    cappedReluOutput = min(1, reluOutput);
    predictedTargets = 2 * cappedReluOutput - 1;
    outputError = predictedTargets - originalTrainingTargets;

    % Backpropagate mean squared error through the 2*x-1 scale and capped ReLU.
    activeGradient = (linearOutput > 0) & (linearOutput < 1);
    linearGradient = (4 / numel(outputError)) * outputError .* activeGradient;
    network.weights = network.weights - learningRate * ...
        (wrappedTrainingInputs' * linearGradient);
    network.bias = network.bias - learningRate * sum(linearGradient, 1);
end

finalLinearOutput = wrappedTrainingInputs * network.weights + ...
    repmat(reshape(network.bias, 1, N), trainingExampleCount, 1);
finalPrediction = 2 * min(1, max(0, finalLinearOutput)) - 1;
network.trainingRMSE = sqrt(mean((finalPrediction(:) - ...
    originalTrainingTargets(:)).^2));
network.epochs = maxEpochs;
end