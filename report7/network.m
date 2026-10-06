% NETWORK.M
% Train a single-layer neural network to reconstruct sine waves from their
% wrapped samples, then plot the reconstruction for one sine wave.
% Change N, frequencyHz, durationSeconds, and wrapThreshold as desired.
% The output uses capped ReLU ([0,1]) followed by 2*x - 1.

clear;
clc;

%% Configurable signal and network parameters
N = 64;                    % Samples/inputs/outputs per wave
frequencyHz = 1;            % Sine-wave frequency in hertz
durationSeconds = 2;        % Duration represented by each wave
wrapThreshold = 0.5;         % Wrapped inputs lie in [-threshold, threshold)
numTrainingExamples = 64;   % Sine waves with different starting phases
learningRate = 0.2;
maxEpochs = 3000;

if ~isscalar(N) || isnan(N) || isinf(N) || N < 2 || N ~= floor(N)
    error('N must be an integer greater than or equal to 2.');
end
if ~isscalar(frequencyHz) || isnan(frequencyHz) || ...
        isinf(frequencyHz) || frequencyHz <= 0
    error('frequencyHz must be a finite, positive scalar.');
end
if ~isscalar(durationSeconds) || isnan(durationSeconds) || ...
        isinf(durationSeconds) || durationSeconds <= 0
    error('durationSeconds must be a finite, positive scalar.');
end
if ~isscalar(wrapThreshold) || isnan(wrapThreshold) || ...
        isinf(wrapThreshold) || wrapThreshold <= 0
    error('wrapThreshold must be a finite, positive scalar.');
end
if ~isscalar(numTrainingExamples) || isnan(numTrainingExamples) || ...
        isinf(numTrainingExamples) || numTrainingExamples < 1 || ...
        numTrainingExamples ~= floor(numTrainingExamples)
    error('numTrainingExamples must be a positive integer.');
end

%% Build training pairs, each containing exactly N equidistant samples
timeSeconds = linspace(0, durationSeconds, N);
wrappedTrainingInputs = zeros(numTrainingExamples, N);
originalTrainingTargets = zeros(numTrainingExamples, N);

for example = 1:numTrainingExamples
    phase = 2 * pi * (example - 1) / numTrainingExamples;
    originalWave = sin(2 * pi * frequencyHz * timeSeconds + phase);
    wrappedWave = mod(originalWave + wrapThreshold, 2 * wrapThreshold) - wrapThreshold;
    wrappedTrainingInputs(example, :) = wrappedWave;
    originalTrainingTargets(example, :) = originalWave;
end

%% Build and train the configurable N-input/N-output network
network = buildNetwork(N);
network = trainReconstructionNetwork(network, wrappedTrainingInputs, ...
    originalTrainingTargets, learningRate, maxEpochs);

%% Reconstruct the zero-phase sine wave from its N wrapped samples
originalSignal = sin(2 * pi * frequencyHz * timeSeconds);
wrappedSignal = mod(originalSignal + wrapThreshold, 2 * wrapThreshold) - wrapThreshold;
reconstructedSignal = applyNetwork(network, wrappedSignal);
reconstructionRMSE = sqrt(mean((reconstructedSignal - originalSignal).^2));

fprintf('Network: %d inputs, one dense layer, %d outputs\n', N, N);
fprintf('Training examples: %d; epochs: %d\n', numTrainingExamples, network.epochs);
fprintf('Training RMSE: %.6f; plotted-wave RMSE: %.6f\n', ...
    network.trainingRMSE, reconstructionRMSE);

%% Plot the target, wrapped input, and network reconstruction
figure;
plot(timeSeconds, originalSignal, 'k--', 'LineWidth', 1.5);
hold on;
plot(timeSeconds, wrappedSignal, 'Color', [0.65, 0.65, 0.65], 'LineWidth', 1.2);
plot(timeSeconds, reconstructedSignal, 'b-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Time (seconds)');
ylabel('Amplitude');
title(sprintf('Sine reconstruction: N = %d, frequency = %.3g Hz', ...
    N, frequencyHz));
legend('Original sine', 'Wrapped input', 'Network reconstruction', ...
    'Location', 'northeast');
xlim([0, durationSeconds]);
ylim([-1.1, 1.1]);