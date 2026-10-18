% NETWORK.M
% Train a two-layer neural network to reconstruct sine waves from their
% wrapped samples, verify on unseen phases, and export a verification plot.
% Change N, frequencyHz, durationSeconds, and wrapThreshold as desired.
% The hidden layer uses tanh; capped ReLU maps outputs to [-1, 1].

clear;
clc;

%% Configurable signal and network parameters
N = 32;                     % Samples/inputs/outputs per wave
frequencyHz = 1;            % Sine-wave frequency in hertz
durationSeconds = 2;        % Duration represented by each wave
wrapThreshold = 0.5;        % Wrapped inputs lie in [-threshold, threshold)
numTrainingExamples = 256;  % Sine waves with different starting phases
learningRate = 0.7;
maxEpochs = 4000 * 4;
earlyStoppingPatience = 500;
minimumValidationImprovement = 1e-6;

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

%% Build separate validation and verification pairs between training phases
numValidationExamples = numTrainingExamples;
numVerificationExamples = numTrainingExamples;
wrappedValidationInputs = zeros(numValidationExamples, N);
originalValidationTargets = zeros(numValidationExamples, N);
wrappedVerificationInputs = zeros(numVerificationExamples, N);
originalVerificationTargets = zeros(numVerificationExamples, N);

for example = 1:numValidationExamples
    phase = 2 * pi * (example - 1 + 1 / 3) / numTrainingExamples;
    originalWave = sin(2 * pi * frequencyHz * timeSeconds + phase);
    wrappedWave = mod(originalWave + wrapThreshold, 2 * wrapThreshold) - wrapThreshold;
    wrappedValidationInputs(example, :) = wrappedWave;
    originalValidationTargets(example, :) = originalWave;
end

for example = 1:numVerificationExamples
    phase = 2 * pi * (example - 1 + 2 / 3) / numTrainingExamples;
    originalWave = sin(2 * pi * frequencyHz * timeSeconds + phase);
    wrappedWave = mod(originalWave + wrapThreshold, 2 * wrapThreshold) - wrapThreshold;
    wrappedVerificationInputs(example, :) = wrappedWave;
    originalVerificationTargets(example, :) = originalWave;
end

%% Build and train the configurable N-input/N-output network
network = buildNetwork(N);
network = trainReconstructionNetwork(network, wrappedTrainingInputs, ...
    originalTrainingTargets, learningRate, maxEpochs, wrappedValidationInputs, ...
    originalValidationTargets, earlyStoppingPatience, minimumValidationImprovement);

%% Verify on held-out phases and select one for the plot
verificationPredictions = applyNetwork(network, wrappedVerificationInputs);
verificationRMSE = sqrt(mean((verificationPredictions(:) - ...
    originalVerificationTargets(:)).^2));
verificationExample = 1;
originalSignal = originalVerificationTargets(verificationExample, :);
wrappedSignal = wrappedVerificationInputs(verificationExample, :);
reconstructedSignal = verificationPredictions(verificationExample, :);

fprintf('Network: %d inputs, one hidden layer with %d units, %d outputs\n', ...
    N, network.hiddenUnits, N);
fprintf('Training examples: %d; validation examples: %d\n', ...
    numTrainingExamples, numValidationExamples);
if network.stoppedEarly
    earlyStoppingText = 'yes';
else
    earlyStoppingText = 'no';
end
fprintf('Epochs: %d / %d; best validation epoch: %d; stopped early: %s\n', ...
    network.epochs, maxEpochs, network.bestEpoch, ...
    earlyStoppingText);
fprintf('Best validation RMSE: %.6f\n', network.validationRMSE);
fprintf('Training RMSE: %.6f\n', network.trainingRMSE);
fprintf('Verification examples: %d; verification RMSE: %.6f\n', ...
    numVerificationExamples, verificationRMSE);

%% Plot a held-out target, wrapped input, and network reconstruction
figure;
plot(timeSeconds, originalSignal, 'k--', 'LineWidth', 1.5);
hold on;
plot(timeSeconds, wrappedSignal, 'Color', [0.65, 0.65, 0.65], 'LineWidth', 1.2);
plot(timeSeconds, reconstructedSignal, 'b-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Time (seconds)');
ylabel('Amplitude');
title(sprintf('Sine verification: N = %d, frequency = %.3g Hz', ...
    N, frequencyHz));
legend('Verification target', 'Wrapped input', 'Network reconstruction', ...
    'Location', 'northeast');
xlim([0, durationSeconds]);
ylim([-1.1, 1.1]);
scriptDirectory = fileparts(mfilename('fullpath'));
svgPath = fullfile(scriptDirectory, 'reconstruction.svg');
print(gcf, svgPath, '-dsvg');

%% Plot training RMSE across epochs and export it as SVG
figure;
plot(network.epochHistory, network.trainingRMSEHistory, 'b-', 'LineWidth', 1.2);
hold on;
plot(network.bestEpoch, network.trainingRMSEHistory(network.bestEpoch), ...
    'ro', 'MarkerFaceColor', 'r');
hold off;
grid on;
xlabel('Epoch');
ylabel('Training RMSE');
title('Training RMSE by epoch');
legend('Training RMSE', 'Best validation checkpoint', 'Location', 'northeast');
trainingRMSEPath = fullfile(scriptDirectory, 'trainingrmse.svg');
print(gcf, trainingRMSEPath, '-dsvg');
