function network = buildNetwork(N)
%BUILDNETWORK Create an N-input/N-output network with a 2N-unit hidden layer.
% The hidden layer uses tanh; the output uses capped ReLU scaled to [-1, 1].

if ~isnumeric(N) || ~isreal(N) || ~isscalar(N) || ...
        isnan(N) || isinf(N) || N < 1 || N ~= floor(N)
    error('N must be a finite positive integer.');
end
% Fixed initialization makes the trained model reproducible without
% changing the caller's random-number generator state.
previousRngState = rng;
rng(1, 'twister');
network.hiddenUnits = 2 * N;
network.inputHiddenWeights = sqrt(2 / N) * randn(N, network.hiddenUnits);
network.hiddenOutputWeights = sqrt(1 / network.hiddenUnits) * ...
    randn(network.hiddenUnits, N);
rng(previousRngState);
network.hiddenBias = zeros(1, network.hiddenUnits);
network.outputBias = 0.5 * ones(1, N);
network.N = N;
end
