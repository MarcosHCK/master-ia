function network = buildNetwork(inputSize, surfaceResolution, hiddenUnits, initialHeight, seed)
%BUILDNETWORK Initialize an image-to-height-map regression MLP.

if nargin < 1 || isempty(inputSize)
    inputSize = 16;
end
if nargin < 2 || isempty(surfaceResolution)
    surfaceResolution = 16;
end
if nargin < 3 || isempty(hiddenUnits)
    hiddenUnits = 48;
end
if nargin < 4 || isempty(initialHeight)
    initialHeight = 0.5;
end
if nargin < 5 || isempty(seed)
    seed = 1;
end
if ~isscalar(inputSize) || inputSize < 2 || inputSize ~= floor(inputSize) || ...
        ~isscalar(surfaceResolution) || surfaceResolution < 2 || ...
        surfaceResolution ~= floor(surfaceResolution) || ...
        ~isscalar(hiddenUnits) || hiddenUnits < 1 || hiddenUnits ~= floor(hiddenUnits) || ...
        ~isscalar(initialHeight) || ~isfinite(initialHeight) || ...
        initialHeight < 0 || initialHeight > 1 || ...
        ~isscalar(seed) || ~isfinite(seed) || seed ~= floor(seed)
    error('Invalid input size, surface resolution, hidden units, height, or seed.');
end

previousRngState = rng;
rng(seed, 'twister');
restoreRng = onCleanup(@() rng(previousRngState));

inputFeatures = inputSize ^ 2;
outputFeatures = surfaceResolution ^ 2;
network.inputSize = inputSize;
network.surfaceResolution = surfaceResolution;
network.hiddenUnits = hiddenUnits;
network.inputMean = 0.5;
network.W1 = sqrt(2 / inputFeatures) * randn(inputFeatures, hiddenUnits);
network.b1 = zeros(1, hiddenUnits);
network.W2 = sqrt(2 / hiddenUnits) * randn(hiddenUnits, outputFeatures);
initialLogit = log(max(initialHeight, 1e-4) / max(1 - initialHeight, 1e-4));
network.b2 = initialLogit * ones(1, outputFeatures);
network.heightLoss = 'mean_squared_error';

clear restoreRng;
end