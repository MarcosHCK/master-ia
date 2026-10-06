function network = buildNetwork(N)
%BUILDNETWORK Create an N-input/N-output, single-layer ReLU network.
% Capped ReLU limits activations to [0, 1], then y = 2*x - 1 scales them
% to the output range [-1, 1].

if ~isnumeric(N) || ~isreal(N) || ~isscalar(N) || ...
        isnan(N) || isinf(N) || N < 1 || N ~= floor(N)
    error('N must be a finite positive integer.');
end
% Fixed initialization makes the trained model reproducible without
% changing the caller's random-number generator state.
previousRngState = rng;
rng(1, 'twister');
network.weights = 0.01 * randn(N, N);
rng(previousRngState);
network.bias = 0.5 * ones(1, N);
network.N = N;
end