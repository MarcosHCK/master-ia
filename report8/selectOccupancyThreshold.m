function [bestThreshold, bestIoU] = selectOccupancyThreshold(network, inputs, targets)
%SELECTOCCUPANCYTHRESHOLD Choose a voxel threshold by validation-set IoU.
% Select on validation data only; reserve test examples for final evaluation.

if ~isnumeric(inputs) || ~isreal(inputs) || ndims(inputs) ~= 2 || ...
        size(inputs, 2) ~= network.inputSize ^ 2 || isempty(inputs) || ...
        any(~isfinite(inputs(:))) || any(inputs(:) < 0) || any(inputs(:) > 1)
    error('inputs must be finite grayscale rows matching network.inputSize.');
end
if ~isnumeric(targets) || ~isreal(targets) || ndims(targets) ~= 2 || ...
        size(targets, 1) ~= size(inputs, 1) || ...
        size(targets, 2) ~= network.voxelResolution ^ 3 || ...
        any(~isfinite(targets(:))) || any((targets(:) ~= 0) & (targets(:) ~= 1))
    error('targets must be matching binary voxel rows.');
end

probabilities = applyNetwork(network, inputs);
thresholds = 0.01:0.01:0.5;
bestThreshold = thresholds(1);
bestIoU = -Inf;
for threshold = thresholds
    predictions = probabilities >= threshold;
    intersections = sum(predictions & targets, 2);
    unions = sum(predictions | targets, 2);
    perExampleIoU = ones(size(unions));
    nonempty = unions > 0;
    perExampleIoU(nonempty) = intersections(nonempty) ./ unions(nonempty);
    currentIoU = mean(perExampleIoU);
    if currentIoU > bestIoU
        bestIoU = currentIoU;
        bestThreshold = threshold;
    end
end
end