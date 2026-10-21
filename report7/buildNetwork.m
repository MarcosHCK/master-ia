function network = buildNetwork(N)
%BUILDNETWORK Create a sparse N-to-4N-to-N reconstruction network.
% The 4N hidden neurons are grouped in blocks of four. Each block maps
% corresponding positions from every non-overlapping four-input window.
% Hidden activations use tanh; the fully connected output uses capped ReLU.

if ~isnumeric(N) || ~isreal(N) || ~isscalar(N) || ...
        isnan(N) || isinf(N) || N < 1 || N ~= floor(N)
    error('N must be a finite positive integer.');
end
if mod(N, 4) ~= 0
    error('N must be divisible by 4 for the grouped input connections.');
end
% Fixed initialization makes the trained model reproducible without
% changing the caller's random-number generator state.
previousRngState = rng;
rng(1, 'twister');
network.hiddenUnits = 4 * N;
network.inputHiddenMask = false(N, network.hiddenUnits);
for hiddenGroup = 1:N
    for positionInWindow = 1:4
        hiddenNeuron = 4 * (hiddenGroup - 1) + positionInWindow;
        inputIndices = positionInWindow:4:N;
        network.inputHiddenMask(inputIndices, hiddenNeuron) = true;
    end
end
fanIn = N / 4;
network.inputHiddenWeights = sqrt(2 / fanIn) * ...
    randn(N, network.hiddenUnits) .* network.inputHiddenMask;
network.hiddenOutputWeights = sqrt(1 / network.hiddenUnits) * ...
    randn(network.hiddenUnits, N);
rng(previousRngState);
network.hiddenBias = zeros(1, network.hiddenUnits);
network.outputBias = 0.5 * ones(1, N);
network.N = N;
end