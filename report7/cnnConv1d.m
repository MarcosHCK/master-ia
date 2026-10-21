function output = cnnConv1d(input, weights, bias, dilation)
%CNNCONV1D Same-padded 1-D convolution over sequence dimension 2.
% Input shape is examples x sequence positions x input channels.

batchSize = size(input, 1);
sequenceLength = size(input, 2);
inputChannels = size(input, 3);
kernelSize = size(weights, 1);
outputChannels = size(weights, 3);
if mod(kernelSize, 2) ~= 1
    error('cnnConv1d requires an odd kernel size.');
end

output = repmat(reshape(bias, 1, 1, outputChannels), ...
    batchSize, sequenceLength, 1);
center = (kernelSize + 1) / 2;
for kernelIndex = 1:kernelSize
    offset = (kernelIndex - center) * dilation;
    outputPositions = max(1, 1 - offset):min(sequenceLength, sequenceLength - offset);
    sourcePositions = outputPositions + offset;
    if ~isempty(outputPositions)
        inputSlice = reshape(input(:, sourcePositions, :), ...
            batchSize * numel(sourcePositions), inputChannels);
        kernelSlice = reshape(weights(kernelIndex, :, :), inputChannels, outputChannels);
        contribution = inputSlice * kernelSlice;
        output(:, outputPositions, :) = output(:, outputPositions, :) + ...
            reshape(contribution, batchSize, numel(outputPositions), outputChannels);
    end
end
end