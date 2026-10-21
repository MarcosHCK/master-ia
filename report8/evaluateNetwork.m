function [metrics, probabilities] = evaluateNetwork(network, inputs, targets)
%EVALUATENETWORK Calculate BCE, voxel IoU, accuracy, and silhouette IoU.

if ~isnumeric(inputs) || ~isreal(inputs) || ndims(inputs) ~= 2 || ...
        size(inputs, 2) ~= network.inputSize ^ 2 || isempty(inputs) || ...
        any(~isfinite(inputs(:))) || any(inputs(:) < 0) || any(inputs(:) > 1)
    error('inputs must be finite grayscale rows with network.inputSize^2 columns.');
end
if ~isnumeric(targets) || ~isreal(targets) || ndims(targets) ~= 2 || ...
        size(targets, 1) ~= size(inputs, 1) || ...
        size(targets, 2) ~= network.voxelResolution ^ 3 || ...
        any(~isfinite(targets(:))) || any((targets(:) ~= 0) & (targets(:) ~= 1))
    error('targets must be matching binary voxel rows.');
end

probabilities = applyNetwork(network, inputs);
crossEntropy = max(-log(max(probabilities, realmin)), 0) .* targets + ...
    max(-log(max(1 - probabilities, realmin)), 0) .* (1 - targets);
metrics.binaryCrossEntropy = mean(sum(crossEntropy, 2));
predictedOccupancy = probabilities >= network.occupancyThreshold;
voxelIntersection = sum(predictedOccupancy & targets, 2);
voxelUnion = sum(predictedOccupancy | targets, 2);
metrics.voxelIoU = mean(safeRatio(voxelIntersection, voxelUnion));
metrics.voxelAccuracy = mean(predictedOccupancy(:) == logical(targets(:)));

sampleCount = size(inputs, 1);
resolution = network.voxelResolution;
projectionIntersection = zeros(sampleCount, 1);
projectionUnion = zeros(sampleCount, 1);
for example = 1:sampleCount
    predictedVolume = reshape(predictedOccupancy(example, :), ...
        resolution, resolution, resolution);
    targetVolume = reshape(targets(example, :), resolution, resolution, resolution);
    predictedSilhouette = any(predictedVolume, 3);
    targetSilhouette = any(targetVolume, 3);
    projectionIntersection(example) = sum(predictedSilhouette(:) & targetSilhouette(:));
    projectionUnion(example) = sum(predictedSilhouette(:) | targetSilhouette(:));
end
metrics.silhouetteIoU = mean(safeRatio(projectionIntersection, projectionUnion));
metrics.numExamples = sampleCount;
metrics.occupancyThreshold = network.occupancyThreshold;
end

function ratio = safeRatio(numerator, denominator)
ratio = ones(size(numerator));
nonempty = denominator > 0;
ratio(nonempty) = numerator(nonempty) ./ denominator(nonempty);
end