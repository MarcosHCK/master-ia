function [prediction, cache] = surface_cnn_forward(network, inputImages)
%SURFACE_CNN_FORWARD Forward pass for normalized grayscale intensity images.

batchSize = size(inputImages, 4);
cache.input = reshape(inputImages, network.N, network.N, 1, batchSize);
cache.stemLinear = surface_cnn_conv2d(cache.input, ...
    network.stemWeights, network.stemBias, 1);
cache.stemActivation = max(0, cache.stemLinear);
cache.blockInput = cache.stemActivation;
cache.blockFirstLinear = surface_cnn_conv2d(cache.blockInput, ...
    network.blocks(1).conv1Weights, network.blocks(1).conv1Bias, 1);
cache.blockFirstActivation = max(0, cache.blockFirstLinear);
cache.blockSecondLinear = surface_cnn_conv2d(cache.blockFirstActivation, ...
    network.blocks(1).conv2Weights, network.blocks(1).conv2Bias, 1);
cache.blockResidualSum = cache.blockInput + cache.blockSecondLinear;
cache.blockOutput = max(0, cache.blockResidualSum);
cache.headLinear = surface_cnn_conv2d(cache.blockOutput, ...
    network.headWeights, network.headBias, 1);
cache.unclippedPrediction = cache.input + ...
    network.residualScale * cache.headLinear;
prediction = min(1, max(0, cache.unclippedPrediction));
end