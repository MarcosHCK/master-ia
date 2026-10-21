function [inputs, targets] = generateSyntheticData(numExamples, inputSize, voxelResolution, seed)
%GENERATESYNTHETICDATA Make grayscale projections paired with voxel solids.
% The orthographic input is the mean occupancy along the viewing (z) axis;
% consequently, pixel brightness retains a coarse measure of object depth.

if nargin < 4 || isempty(seed)
    seed = 1;
end
if ~isscalar(numExamples) || numExamples < 1 || numExamples ~= floor(numExamples) || ...
        ~isscalar(inputSize) || inputSize < 2 || inputSize ~= floor(inputSize) || ...
        ~isscalar(voxelResolution) || voxelResolution < 4 || ...
        voxelResolution ~= floor(voxelResolution) || ...
        ~isscalar(seed) || ~isfinite(seed) || seed ~= floor(seed)
    error('Counts and resolutions must be integers; the seed must be a finite integer.');
end

previousRngState = rng;
rng(seed, 'twister');
restoreRng = onCleanup(@() rng(previousRngState));

coordinates = linspace(-1, 1, voxelResolution);
[worldX, worldY, worldZ] = ndgrid(coordinates, coordinates, coordinates);
inputs = zeros(numExamples, inputSize * inputSize);
targets = zeros(numExamples, voxelResolution ^ 3);

for example = 1:numExamples
    shapeType = randi(5);
    center = 0.18 * (2 * rand(1, 3) - 1);
    angles = (pi / 6) * (2 * rand(1, 3) - 1);
    rotationX = [1, 0, 0; 0, cos(angles(1)), -sin(angles(1)); ...
        0, sin(angles(1)), cos(angles(1))];
    rotationY = [cos(angles(2)), 0, sin(angles(2)); 0, 1, 0; ...
        -sin(angles(2)), 0, cos(angles(2))];
    rotationZ = [cos(angles(3)), -sin(angles(3)), 0; ...
        sin(angles(3)), cos(angles(3)), 0; 0, 0, 1];
    rotation = rotationZ * rotationY * rotationX;

    offsetX = worldX - center(1);
    offsetY = worldY - center(2);
    offsetZ = worldZ - center(3);
    localX = rotation(1, 1) * offsetX + rotation(2, 1) * offsetY + ...
        rotation(3, 1) * offsetZ;
    localY = rotation(1, 2) * offsetX + rotation(2, 2) * offsetY + ...
        rotation(3, 2) * offsetZ;
    localZ = rotation(1, 3) * offsetX + rotation(2, 3) * offsetY + ...
        rotation(3, 3) * offsetZ;

    switch shapeType
        case 1  % Sphere
            radius = 0.42 + 0.22 * rand();
            occupied = localX .^ 2 + localY .^ 2 + localZ .^ 2 <= radius ^ 2;
        case 2  % Ellipsoid
            radii = 0.38 + 0.25 * rand(1, 3);
            occupied = (localX / radii(1)) .^ 2 + ...
                (localY / radii(2)) .^ 2 + (localZ / radii(3)) .^ 2 <= 1;
        case 3  % Rectangular prism
            halfWidths = 0.34 + 0.25 * rand(1, 3);
            occupied = abs(localX) <= halfWidths(1) & ...
                abs(localY) <= halfWidths(2) & abs(localZ) <= halfWidths(3);
        case 4  % Elliptical cylinder, with axis along local z
            radii = 0.34 + 0.14 * rand(1, 2);
            halfHeight = 0.42 + 0.20 * rand();
            occupied = (localX / radii(1)) .^ 2 + ...
                (localY / radii(2)) .^ 2 <= 1 & abs(localZ) <= halfHeight;
        otherwise  % Cone, with tip along local positive z
            baseRadii = 0.42 + 0.12 * rand(1, 2);
            halfHeight = 0.48 + 0.16 * rand();
            normalizedHeight = (localZ + halfHeight) / (2 * halfHeight);
            radiusFactor = max(0, 1 - normalizedHeight);
            occupied = normalizedHeight >= 0 & normalizedHeight <= 1 & ...
                (localX / baseRadii(1)) .^ 2 + ...
                (localY / baseRadii(2)) .^ 2 <= radiusFactor .^ 2;
    end

    volume = double(occupied);
    projection = mean(volume, 3);
    image = resizeGrayscaleImage(projection, inputSize, inputSize);
    inputs(example, :) = image(:)';
    targets(example, :) = volume(:)';
end

clear restoreRng;
end