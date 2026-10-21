% NETWORK.M
% Train a residual 1-D CNN to reconstruct sine waves from their
% wrapped samples, verify on unseen phases/frequencies, and export plots.
% Change N, frequencyHz, frequencyVariationHz, durationSeconds, and wrapThreshold.
% The hidden layer uses tanh; capped ReLU maps outputs to [-1, 1].

clear;
clc;

%% Configurable signal and network parameters
N = 64;                     % Samples/inputs/outputs per wave
modelType = 'cnn';           % 'cnn' or 'sparse'
frequencyHz = 1;            % Sine-wave frequency in hertz
frequencyVariationHz = 0.5; % Frequencies span frequencyHz +/- this amount
durationSeconds = 2;        % Duration represented by each wave
wrapThreshold = 0.5;        % Wrapped inputs lie in [-threshold, threshold)
numFrequencyLevels = 16;
numPhasesPerFrequency = 16;
numTrainingExamples = numFrequencyLevels * numPhasesPerFrequency;
learningRate = 0.7;
maxEpochs = 4000 * 4;
earlyStoppingPatience = 500;
minimumValidationImprovement = 1e-6;
minimumFrequencyHz = frequencyHz - frequencyVariationHz;
maximumFrequencyHz = frequencyHz + frequencyVariationHz;

if ~isscalar(N) || isnan(N) || isinf(N) || N < 2 || N ~= floor(N)
    error('N must be an integer greater than or equal to 2.');
end
if ~isscalar(frequencyHz) || ~isreal(frequencyHz) || isnan(frequencyHz) || ...
        isinf(frequencyHz) || frequencyHz <= 0
    error('frequencyHz must be a finite, positive scalar.');
end
if ~isscalar(frequencyVariationHz) || ~isreal(frequencyVariationHz) || ...
        isnan(frequencyVariationHz) || isinf(frequencyVariationHz) || ...
        frequencyVariationHz <= 0 || minimumFrequencyHz <= 0
    error('frequencyVariationHz must be positive and keep all frequencies positive.');
end
if ~isscalar(durationSeconds) || isnan(durationSeconds) || ...
        isinf(durationSeconds) || durationSeconds <= 0
    error('durationSeconds must be a finite, positive scalar.');
end
if ~isscalar(wrapThreshold) || isnan(wrapThreshold) || ...
        isinf(wrapThreshold) || wrapThreshold <= 0
    error('wrapThreshold must be a finite, positive scalar.');
end
if ~isscalar(numFrequencyLevels) || numFrequencyLevels < 1 || ...
        numFrequencyLevels ~= floor(numFrequencyLevels) || ...
        ~isscalar(numPhasesPerFrequency) || numPhasesPerFrequency < 1 || ...
        numPhasesPerFrequency ~= floor(numPhasesPerFrequency)
    error('Frequency-level and phase counts must be positive integers.');
end

%% Build training pairs over a grid of frequencies and starting phases
timeSeconds = linspace(0, durationSeconds, N);
wrappedTrainingInputs = zeros(numTrainingExamples, N);
originalTrainingTargets = zeros(numTrainingExamples, N);

for example = 1:numTrainingExamples
    frequencyLevel = ceil(example / numPhasesPerFrequency);
    phaseLevel = mod(example - 1, numPhasesPerFrequency);
    frequencyGridIndex = 3 * (frequencyLevel - 1) + 1;
    signalFrequencyHz = minimumFrequencyHz + ...
        (frequencyGridIndex - 1) * (maximumFrequencyHz - minimumFrequencyHz) / ...
        (3 * numFrequencyLevels - 1);
    phase = 2 * pi * phaseLevel / numPhasesPerFrequency;
    originalWave = sin(2 * pi * signalFrequencyHz * timeSeconds + phase);
    wrappedWave = mod(originalWave + wrapThreshold, 2 * wrapThreshold) - wrapThreshold;
    wrappedTrainingInputs(example, :) = wrappedWave;
    originalTrainingTargets(example, :) = originalWave;
end

%% Build separate frequency/phase grids for validation and verification
numValidationExamples = numTrainingExamples;
numVerificationExamples = numTrainingExamples;
wrappedValidationInputs = zeros(numValidationExamples, N);
originalValidationTargets = zeros(numValidationExamples, N);
wrappedVerificationInputs = zeros(numVerificationExamples, N);
originalVerificationTargets = zeros(numVerificationExamples, N);
verificationFrequenciesHz = zeros(numVerificationExamples, 1);

for example = 1:numValidationExamples
    frequencyLevel = ceil(example / numPhasesPerFrequency);
    phaseLevel = mod(example - 1, numPhasesPerFrequency);
    frequencyGridIndex = 3 * (frequencyLevel - 1) + 2;
    signalFrequencyHz = minimumFrequencyHz + ...
        (frequencyGridIndex - 1) * (maximumFrequencyHz - minimumFrequencyHz) / ...
        (3 * numFrequencyLevels - 1);
    phase = 2 * pi * (phaseLevel + 1 / 3) / numPhasesPerFrequency;
    originalWave = sin(2 * pi * signalFrequencyHz * timeSeconds + phase);
    wrappedWave = mod(originalWave + wrapThreshold, 2 * wrapThreshold) - wrapThreshold;
    wrappedValidationInputs(example, :) = wrappedWave;
    originalValidationTargets(example, :) = originalWave;
end

for example = 1:numVerificationExamples
    frequencyLevel = ceil(example / numPhasesPerFrequency);
    phaseLevel = mod(example - 1, numPhasesPerFrequency);
    frequencyGridIndex = 3 * (frequencyLevel - 1) + 3;
    signalFrequencyHz = minimumFrequencyHz + ...
        (frequencyGridIndex - 1) * (maximumFrequencyHz - minimumFrequencyHz) / ...
        (3 * numFrequencyLevels - 1);
    phase = 2 * pi * (phaseLevel + 2 / 3) / numPhasesPerFrequency;
    originalWave = sin(2 * pi * signalFrequencyHz * timeSeconds + phase);
    wrappedWave = mod(originalWave + wrapThreshold, 2 * wrapThreshold) - wrapThreshold;
    wrappedVerificationInputs(example, :) = wrappedWave;
    originalVerificationTargets(example, :) = originalWave;
    verificationFrequenciesHz(example) = signalFrequencyHz;
end

%% Build and train the selected network with shared data and settings
if strcmp(modelType, 'cnn')
    network = buildCNNNetwork(N);
    network = trainCNNNetwork(network, wrappedTrainingInputs, ...
        originalTrainingTargets, learningRate, maxEpochs, wrappedValidationInputs, ...
        originalValidationTargets, earlyStoppingPatience, minimumValidationImprovement);
elseif strcmp(modelType, 'sparse')
    network = buildNetwork(N);
    network = trainReconstructionNetwork(network, wrappedTrainingInputs, ...
        originalTrainingTargets, learningRate, maxEpochs, wrappedValidationInputs, ...
        originalValidationTargets, earlyStoppingPatience, minimumValidationImprovement);
else
    error('modelType must be either ''cnn'' or ''sparse''.');
end

%% Verify on held-out phases and select one for the plot
if strcmp(modelType, 'cnn')
    verificationPredictions = applyCNNNetwork(network, wrappedVerificationInputs);
else
    verificationPredictions = applyNetwork(network, wrappedVerificationInputs);
end
verificationRMSE = sqrt(mean((verificationPredictions(:) - ...
    originalVerificationTargets(:)).^2));
verificationExample = 1;
verificationFrequencyHz = verificationFrequenciesHz(verificationExample);
originalSignal = originalVerificationTargets(verificationExample, :);
wrappedSignal = wrappedVerificationInputs(verificationExample, :);
reconstructedSignal = verificationPredictions(verificationExample, :);

if strcmp(modelType, 'cnn')
    fprintf('Network: residual 1-D CNN, %d channels, dilations %s\n', ...
        network.channels, mat2str(network.dilations));
else
    fprintf('Network: %d -> %d -> %d units (sparse input connections)\n', ...
        N, network.hiddenUnits, N);
end
fprintf('Frequency range: %.3f-%.3f Hz; %d levels x %d phases per split\n', ...
    minimumFrequencyHz, maximumFrequencyHz, ...
    numFrequencyLevels, numPhasesPerFrequency);
fprintf('Training examples: %d; validation: %d; verification: %d\n', ...
    numTrainingExamples, numValidationExamples, numVerificationExamples);
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
    N, verificationFrequencyHz));
legend('Verification target', 'Wrapped input', 'Network reconstruction', ...
    'Location', 'northeast');
xlim([0, durationSeconds]);
ylim([-1.1, 1.1]);
scriptDirectory = fileparts(mfilename('fullpath'));
svgPath = fullfile(scriptDirectory, 'reconstruction.svg');
print(gcf, svgPath, '-dsvg');

%% Plot training and validation RMSE across epochs and export as SVG
figure;
plot(network.epochHistory, network.trainingRMSEHistory, 'b-', 'LineWidth', 1.2);
hold on;
plot(network.epochHistory, network.validationRMSEHistory, 'Color', [0.85, 0.33, 0.1], ...
    'LineWidth', 1.2);
plot(network.bestEpoch, network.validationRMSEHistory(network.bestEpoch), ...
    'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold off;
grid on;
xlabel('Epoch');
ylabel('RMSE');
title('Training and validation RMSE by epoch');
legend('Training RMSE', 'Validation RMSE', 'Best validation checkpoint', ...
    'Location', 'northeast');
trainingRMSEPath = fullfile(scriptDirectory, 'trainingrmse.svg');
print(gcf, trainingRMSEPath, '-dsvg');
