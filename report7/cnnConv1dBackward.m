function [inputGradient, weightGradient, biasGradient] = ...
        cnnConv1dBackward(input, weights, outputGradient, dilation)
%CNNCONV1DBACKWARD Backpropagate through cnnConv1d.

batchSize = size(input, 1);
sequenceLength = size(input, 2);
inputChannels = size(input, 3);
kernelSize = size(weights, 1);
outputChannels = size(weights, 3);

inputGradient = zeros(size(input));
weightGradient = zeros(size(weights));
biasGradient = zeros(1, outputChannels);
center = (kernelSize + 1) / 2;
biasGradient = sum(reshape(outputGradient, ...
    batchSize * sequenceLength, outputChannels), 1);

for kernelIndex = 1:kernelSize
    offset = (kernelIndex - center) * dilation;
    outputPositions = max(1, 1 - offset):min(sequenceLength, sequenceLength - offset);
    sourcePositions = outputPositions + offset;
    if ~isempty(outputPositions)
        inputSlice = reshape(input(:, sourcePositions, :), ...
            batchSize * numel(sourcePositions), inputChannels);
        gradientSlice = reshape(outputGradient(:, outputPositions, :), ...
            batchSize * numel(outputPositions), outputChannels);
        kernelSlice = reshape(weights(kernelIndex, :, :), inputChannels, outputChannels);
        weightGradient(kernelIndex, :, :) = reshape( ...
            inputSlice' * gradientSlice, 1, inputChannels, outputChannels);
        inputContribution = gradientSlice * kernelSlice';
        inputGradient(:, sourcePositions, :) = ...
            inputGradient(:, sourcePositions, :) + reshape(inputContribution, ...
            batchSize, numel(sourcePositions), inputChannels);
    end
end
end