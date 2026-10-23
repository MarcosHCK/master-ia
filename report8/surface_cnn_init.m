function network = surface_cnn_init(N, seed)
%SURFACE_CNN_INIT Initialize the compact residual 2-D CNN.
% Inputs and outputs are grayscale intensity maps in [0, 1]. The learned
% residual is added to the input, so the initial network approximates the
% identity map and can learn to remove observation noise.

if nargin < 1 || isempty(N)
    N = 64;
end
if nargin < 2 || isempty(seed)
    seed = 1;
end
if ~isnumeric(N) || ~isscalar(N) || ~surface_isfinite(N) || ...
        N < 8 || N ~= floor(N)
    error('N must be a finite integer greater than or equal to 8.');
end

previousRngState = rng;
rng(seed, 'twister');
channels = 4;
network.formatVersion = 1;
network.N = N;
network.channels = channels;
network.kernelSize = 3;
network.residualScale = 0.25;
network.stemWeights = sqrt(2 / 9) * randn(3, 3, 1, channels);
network.stemBias = zeros(1, channels);
network.blocks(1).conv1Weights = sqrt(2 / (9 * channels)) * ...
    randn(3, 3, channels, channels);
network.blocks(1).conv1Bias = zeros(1, channels);
network.blocks(1).conv2Weights = 0.1 * sqrt(2 / (9 * channels)) * ...
    randn(3, 3, channels, channels);
network.blocks(1).conv2Bias = zeros(1, channels);
network.headWeights = 0.01 * randn(1, 1, channels, 1);
network.headBias = 0;
rng(previousRngState);
end