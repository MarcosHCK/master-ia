function result = applyNetwork(network, imageOrBatch)
%APPLYNETWORK Predict normalized surface height for an image or image batch.
% A single N-by-N grayscale depthmap returns an R-by-R height grid.
% A batch with one flattened image per row returns examples-by-R^2 heights.

if ~isstruct(network) || ~isfield(network, 'inputSize') || ...
        ~isfield(network, 'surfaceResolution') || ~isfield(network, 'hiddenUnits') || ...
        ~isfield(network, 'inputMean') || ~isfield(network, 'W1') || ...
        ~isfield(network, 'b1') || ~isfield(network, 'W2') || ~isfield(network, 'b2')
    error('network must be a height-map model returned by buildNetwork.');
end
inputFeatures = network.inputSize ^ 2;
singleImage = false;
if ismatrix(imageOrBatch) && ...
        isequal(size(imageOrBatch), [network.inputSize, network.inputSize])
    inputs = reshape(double(imageOrBatch), 1, inputFeatures);
    singleImage = true;
elseif isvector(imageOrBatch) && numel(imageOrBatch) == inputFeatures
    inputs = reshape(double(imageOrBatch), 1, inputFeatures);
    singleImage = true;
elseif isnumeric(imageOrBatch) && isreal(imageOrBatch) && ...
        ismatrix(imageOrBatch) && size(imageOrBatch, 2) == inputFeatures
    inputs = double(imageOrBatch);
else
    error('Input must be an N-by-N image, an N^2-vector, or a batch with N^2 columns.');
end
if ~isreal(inputs) || any(~isfinite(inputs(:))) || ...
        any(inputs(:) < 0) || any(inputs(:) > 1)
    error('Depthmap inputs must contain finite values in [0,1].');
end
outputFeatures = network.surfaceResolution ^ 2;
if ~isequal(size(network.W1), [inputFeatures, network.hiddenUnits]) || ...
        numel(network.b1) ~= network.hiddenUnits || ...
        ~isequal(size(network.W2), [network.hiddenUnits, outputFeatures]) || ...
        numel(network.b2) ~= outputFeatures
    error('Network parameter dimensions do not match its configured layer sizes.');
end

hiddenActivations = max(0, (inputs - network.inputMean) * network.W1 + ...
    repmat(network.b1, size(inputs, 1), 1));
logits = hiddenActivations * network.W2 + ...
    repmat(network.b2, size(inputs, 1), 1);
logits = min(50, max(-50, logits));
heights = 1 ./ (1 + exp(-logits));
if singleImage
    result = reshape(heights(1, :), network.surfaceResolution, ...
        network.surfaceResolution);
else
    result = heights;
end
end