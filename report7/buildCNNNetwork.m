function network = buildCNNNetwork(N)
%BUILDCNNNETWORK Create a small residual 1-D CNN for wave reconstruction.
% Architecture: scalar input, a 4-channel convolutional stem, three
% residual blocks with dilation factors 1, 2, and 4, then a scalar head.

if ~isnumeric(N) || ~isreal(N) || ~isscalar(N) || ...
        isnan(N) || isinf(N) || N < 2 || N ~= floor(N)
    error('N must be a finite integer greater than or equal to 2.');
end

channels = 4;
kernelSize = 3;
dilations = [1, 2, 4];

previousRngState = rng;
rng(1, 'twister');
network.N = N;
network.channels = channels;
network.kernelSize = kernelSize;
network.dilations = dilations;
network.maxGradientNorm = 0.1;
network.stemWeights = randn(kernelSize, 1, channels) / sqrt(kernelSize);
network.stemBias = zeros(1, channels);

for block = 1:numel(dilations)
    scale = 0.5 / sqrt(kernelSize * channels);
    network.blocks(block).conv1Weights = ...
        scale * randn(kernelSize, channels, channels);
    network.blocks(block).conv1Bias = zeros(1, channels);
    % Small second-convolution weights keep residual blocks near identity
    % while still allowing gradients to reach earlier layers immediately.
    network.blocks(block).conv2Weights = ...
        0.1 * scale * randn(kernelSize, channels, channels);
    network.blocks(block).conv2Bias = zeros(1, channels);
end

network.outputWeights = 0.1 * randn(1, channels, 1) / sqrt(channels);
network.outputBias = 0.5;
rng(previousRngState);
end