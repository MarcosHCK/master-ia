function [inputs, targets] = generateSyntheticData(numExamples, inputSize, surfaceResolution, seed)
%GENERATESYNTHETICDATA Make top-down depthmaps paired with height fields.
% Input pixels encode normalized surface height; targets are continuous
% normalized heights sampled on the requested output grid.

if nargin < 4 || isempty(seed)
    seed = 1;
end
if ~isscalar(numExamples) || numExamples < 1 || numExamples ~= floor(numExamples) || ...
        ~isscalar(inputSize) || inputSize < 2 || inputSize ~= floor(inputSize) || ...
        ~isscalar(surfaceResolution) || surfaceResolution < 2 || ...
        surfaceResolution ~= floor(surfaceResolution) || ...
        ~isscalar(seed) || ~isfinite(seed) || seed ~= floor(seed)
    error('Counts and resolutions must be integers; the seed must be a finite integer.');
end

previousRngState = rng;
rng(seed, 'twister');
restoreRng = onCleanup(@() rng(previousRngState));

inputs = zeros(numExamples, inputSize * inputSize);
targets = zeros(numExamples, surfaceResolution * surfaceResolution);
for example = 1:numExamples
    % Smooth random field with several localized hills/depressions. The
    % broad Gaussian components make varied, continuous, height-field-like
    % surfaces without requiring image-processing or deep-learning packages.
    [surfaceX, surfaceY] = meshgrid(linspace(-1, 1, surfaceResolution), ...
        linspace(-1, 1, surfaceResolution));
    baseHeight = 0.25 + 0.20 * rand();
    slopeX = 0.12 * (2 * rand() - 1);
    slopeY = 0.12 * (2 * rand() - 1);
    heightMap = baseHeight + slopeX * surfaceX + slopeY * surfaceY;
    featureCount = randi([2, 6]);
    for feature = 1:featureCount
        centerX = 1.5 * rand() - 0.75;
        centerY = 1.5 * rand() - 0.75;
        widthX = 0.12 + 0.48 * rand();
        widthY = 0.12 + 0.48 * rand();
        amplitude = (0.08 + 0.35 * rand()) * (2 * (rand() > 0.5) - 1);
        heightMap = heightMap + amplitude * exp(-0.5 * ...
            (((surfaceX - centerX) / widthX) .^ 2 + ...
            ((surfaceY - centerY) / widthY) .^ 2));
    end

    % Add a small smooth ripple to avoid an overly narrow family of shapes.
    frequency = 1 + 3 * rand();
    phase = 2 * pi * rand();
    heightMap = heightMap + 0.035 * sin(frequency * pi * surfaceX + phase) .* ...
        cos((frequency + 0.5) * pi * surfaceY - phase);
    heightMap = min(1, max(0, heightMap));

    depthmap = resizeGrayscaleImage(heightMap, inputSize, inputSize);
    inputs(example, :) = depthmap(:)';
    targets(example, :) = heightMap(:)';
end

clear restoreRng;
end