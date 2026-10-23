function [inputGradient, weightGradient, biasGradient] = ...
        surface_cnn_conv2d_backward(input, weights, outputGradient, dilation)
%SURFACE_CNN_CONV2D_BACKWARD Backpropagate through surface_cnn_conv2d.

if nargin < 4 || isempty(dilation)
    dilation = 1;
end
[height, width, inputChannels, batchSize] = size(input);
outputChannels = size(weights, 4);
kernelHeight = size(weights, 1);
kernelWidth = size(weights, 2);
inputGradient = zeros(size(input));
weightGradient = zeros(size(weights));
packedOutputGradient = reshape(permute(outputGradient, [1, 2, 4, 3]), ...
    height * width * batchSize, outputChannels);
biasGradient = sum(packedOutputGradient, 1);
centerY = (kernelHeight + 1) / 2;
centerX = (kernelWidth + 1) / 2;

for kernelY = 1:kernelHeight
    offsetY = (kernelY - centerY) * dilation;
    outputRows = max(1, 1 - offsetY):min(height, height - offsetY);
    sourceRows = outputRows + offsetY;
    for kernelX = 1:kernelWidth
        offsetX = (kernelX - centerX) * dilation;
        outputColumns = max(1, 1 - offsetX):min(width, width - offsetX);
        sourceColumns = outputColumns + offsetX;
        inputSlice = input(sourceRows, sourceColumns, :, :);
        packedInput = reshape(permute(inputSlice, [1, 2, 4, 3]), ...
            numel(outputRows) * numel(outputColumns) * batchSize, inputChannels);
        gradientSlice = outputGradient(outputRows, outputColumns, :, :);
        packedGradient = reshape(permute(gradientSlice, [1, 2, 4, 3]), ...
            numel(outputRows) * numel(outputColumns) * batchSize, outputChannels);
        weightGradient(kernelY, kernelX, :, :) = reshape( ...
            packedInput' * packedGradient, 1, 1, inputChannels, outputChannels);
        kernelSlice = reshape(weights(kernelY, kernelX, :, :), ...
            inputChannels, outputChannels);
        inputContribution = packedGradient * kernelSlice';
        inputContribution = permute(reshape(inputContribution, ...
            numel(outputRows), numel(outputColumns), batchSize, inputChannels), ...
            [1, 2, 4, 3]);
        inputGradient(sourceRows, sourceColumns, :, :) = ...
            inputGradient(sourceRows, sourceColumns, :, :) + inputContribution;
    end
end
end