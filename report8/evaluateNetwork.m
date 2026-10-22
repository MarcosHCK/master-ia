function [metrics, heightPredictions] = evaluateNetwork(network, inputs, targets)
%EVALUATENETWORK Calculate height-map regression metrics on normalized data.

if ~isnumeric(inputs) || ~isreal(inputs) || ndims(inputs) ~= 2 || ...
        size(inputs, 2) ~= network.inputSize ^ 2 || isempty(inputs) || ...
        any(~isfinite(inputs(:))) || any(inputs(:) < 0) || any(inputs(:) > 1)
    error('inputs must be finite grayscale rows with network.inputSize^2 columns.');
end
if ~isnumeric(targets) || ~isreal(targets) || ndims(targets) ~= 2 || ...
        size(targets, 1) ~= size(inputs, 1) || ...
        size(targets, 2) ~= network.surfaceResolution ^ 2 || ...
        any(~isfinite(targets(:))) || any(targets(:) < 0) || any(targets(:) > 1)
    error('targets must be matching normalized height-map rows in [0,1].');
end

heightPredictions = applyNetwork(network, inputs);
errors = heightPredictions - targets;
metrics.meanSquaredError = mean(errors(:) .^ 2);
metrics.rootMeanSquaredError = sqrt(metrics.meanSquaredError);
metrics.meanAbsoluteError = mean(abs(errors(:)));
metrics.numExamples = size(inputs, 1);
metrics.surfaceResolution = network.surfaceResolution;
end