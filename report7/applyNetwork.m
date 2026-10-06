function reconstructedSamples = applyNetwork(network, wrappedSamples)
%APPLYNETWORK Reconstruct one or more wrapped waves using a trained network.
% Pass one N-value row/vector, or a matrix with one N-value sample per row.

if ~isstruct(network) || ~isfield(network, 'N') || ...
        ~isfield(network, 'weights') || ~isfield(network, 'bias')
    error('network must be a model returned by buildNetwork.');
end
N = network.N;
if isvector(wrappedSamples) && numel(wrappedSamples) == N
    wrappedSamples = reshape(wrappedSamples, 1, N);
end
if ~isnumeric(wrappedSamples) || ~isreal(wrappedSamples) || ...
        ndims(wrappedSamples) ~= 2 || size(wrappedSamples, 2) ~= N || ...
        any(isnan(wrappedSamples(:))) || any(isinf(wrappedSamples(:)))
    error('wrappedSamples must be a finite vector or matrix with N columns.');
end
if ~isequal(size(network.weights), [N, N]) || numel(network.bias) ~= N
    error('The network weights or bias do not match network.N.');
end

linearOutput = wrappedSamples * network.weights + ...
    repmat(reshape(network.bias, 1, N), size(wrappedSamples, 1), 1);
reluOutput = max(0, linearOutput);
cappedReluOutput = min(1, reluOutput);
reconstructedSamples = 2 * cappedReluOutput - 1;
end