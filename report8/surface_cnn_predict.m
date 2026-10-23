function predictedIntensities = surface_cnn_predict(network, inputImages)
%SURFACE_CNN_PREDICT Denoise one or more square grayscale images.
% Accepts an N-by-N image or N-by-N-by-example array of double intensities
% in [0,1]. Returns the same geometry, with one clean intensity per pixel.

if ~isstruct(network) || ~isfield(network, 'N') || ...
        ~isfield(network, 'stemWeights') || ~isfield(network, 'blocks') || ...
        ~isfield(network, 'headWeights') || ~isfield(network, 'residualScale')
    error('network must be a trained surface CNN structure.');
end
if ~isnumeric(inputImages) || ~isreal(inputImages) || ...
        any(~surface_isfinite(inputImages(:))) || any(inputImages(:) < 0) || ...
        any(inputImages(:) > 1)
    error('inputImages must contain finite grayscale intensities in [0,1].');
end
if ismatrix(inputImages) && isequal(size(inputImages), [network.N, network.N])
    inputImages = reshape(inputImages, network.N, network.N, 1);
end
if ndims(inputImages) > 4 || size(inputImages, 1) ~= network.N || ...
        size(inputImages, 2) ~= network.N || size(inputImages, 3) < 1 || ...
        (ndims(inputImages) == 4 && size(inputImages, 3) ~= 1)
    error('inputImages must have geometry N-by-N or N-by-N-by-example.');
end
inputImages = reshape(double(inputImages), network.N, network.N, 1, []);
[prediction, ~] = surface_cnn_forward(network, inputImages);
predictedIntensities = reshape(prediction, network.N, network.N, []);
if size(predictedIntensities, 3) == 1
    predictedIntensities = predictedIntensities(:, :, 1);
end
end