function network = buildNetwork(inputSize, voxelResolution, hiddenUnits, initialOccupancy, seed)
%BUILDNETWORK Initialize a depthmap-to-voxel-occupancy MLP.

if nargin < 1 || isempty(inputSize)
    inputSize = 16;
end
if nargin < 2 || isempty(voxelResolution)
    voxelResolution = 16;
end
if nargin < 3 || isempty(hiddenUnits)
    hiddenUnits = 48;
end
if nargin < 4 || isempty(initialOccupancy)
    initialOccupancy = 0.2;
end
if nargin < 5 || isempty(seed)
    seed = 1;
end
if ~isscalar(inputSize) || inputSize < 2 || inputSize ~= floor(inputSize) || ...
        ~isscalar(voxelResolution) || voxelResolution < 2 || ...
        voxelResolution ~= floor(voxelResolution) || ...
        ~isscalar(hiddenUnits) || hiddenUnits < 1 || hiddenUnits ~= floor(hiddenUnits) || ...
        ~isscalar(initialOccupancy) || ~isfinite(initialOccupancy) || ...
        initialOccupancy <= 0 || initialOccupancy >= 1 || ...
        ~isscalar(seed) || ~isfinite(seed) || seed ~= floor(seed)
    error('Invalid input size, voxel resolution, hidden units, occupancy, or seed.');
end

previousRngState = rng;
rng(seed, 'twister');
restoreRng = onCleanup(@() rng(previousRngState));

inputFeatures = inputSize ^ 2;
outputFeatures = voxelResolution ^ 3;
network.inputSize = inputSize;
network.voxelResolution = voxelResolution;
network.hiddenUnits = hiddenUnits;
network.inputMean = 0.5;
network.W1 = sqrt(2 / inputFeatures) * randn(inputFeatures, hiddenUnits);
network.b1 = zeros(1, hiddenUnits);
network.W2 = sqrt(2 / hiddenUnits) * randn(hiddenUnits, outputFeatures);
network.b2 = log(initialOccupancy / (1 - initialOccupancy)) * ...
    ones(1, outputFeatures);
network.occupancyThreshold = 0.5;

clear restoreRng;
end