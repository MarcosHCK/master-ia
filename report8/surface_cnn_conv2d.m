function output = surface_cnn_conv2d(input, weights, bias, dilation)
%SURFACE_CNN_CONV2D Same-padded 2-D convolution using MATLAB/Octave arrays.
% Input and output layout is height-by-width-by-channel-by-example. This
% explicit implementation avoids toolbox-specific convolution functions.

if nargin < 4 || isempty(dilation)
    dilation = 1;
end
[height, width, inputChannels, batchSize] = size(input);
kernelHeight = size(weights, 1);
kernelWidth = size(weights, 2);
outputChannels = size(weights, 4);
if mod(kernelHeight, 2) ~= 1 || mod(kernelWidth, 2) ~= 1
    error('surface_cnn_conv2d requires odd kernel dimensions.');
end
if size(weights, 3) ~= inputChannels || numel(bias) ~= outputChannels
    error('Convolution weights, bias, and input channels do not match.');
end

output = repmat(reshape(bias, 1, 1, outputChannels, 1), ...
    height, width, 1, batchSize);
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
        kernelSlice = reshape(weights(kernelY, kernelX, :, :), ...
            inputChannels, outputChannels);
        contribution = packedInput * kernelSlice;
        contribution = permute(reshape(contribution, ...
            numel(outputRows), numel(outputColumns), batchSize, outputChannels), ...
            [1, 2, 4, 3]);
        output(outputRows, outputColumns, :, :) = ...
            output(outputRows, outputColumns, :, :) + contribution;
    end
end
end