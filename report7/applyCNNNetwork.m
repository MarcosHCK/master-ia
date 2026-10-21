function reconstructedSamples = applyCNNNetwork(network, wrappedSamples)
%APPLYCNNNETWORK Apply the residual 1-D CNN to wrapped signals.
% Rows are examples and columns are sequence positions.

if ~isnumeric(wrappedSamples) || ~isreal(wrappedSamples) || ...
        ndims(wrappedSamples) ~= 2 || size(wrappedSamples, 2) ~= network.N || ...
        any(isnan(wrappedSamples(:))) || any(isinf(wrappedSamples(:)))
    error('wrappedSamples must be a finite matrix with network.N columns.');
end

input = reshape(wrappedSamples, size(wrappedSamples, 1), network.N, 1);
features = cnnConv1d(input, network.stemWeights, network.stemBias, 1);
features = max(0, features);
for block = 1:numel(network.dilations)
    residualInput = features;
    firstLinear = cnnConv1d(residualInput, ...
        network.blocks(block).conv1Weights, ...
        network.blocks(block).conv1Bias, network.dilations(block));
    firstActivation = max(0, firstLinear);
    secondLinear = cnnConv1d(firstActivation, ...
        network.blocks(block).conv2Weights, ...
        network.blocks(block).conv2Bias, network.dilations(block));
    features = max(0, residualInput + secondLinear);
end

outputLinear = cnnConv1d(features, network.outputWeights, ...
    network.outputBias, 1);
reconstructedSamples = 2 * min(1, max(0, outputLinear(:, :, 1))) - 1;
end