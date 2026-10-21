% NETWORK.M Train and evaluate a synthetic grayscale-image-to-voxel MLP.
% Run with: octave --no-gui --quiet network.m
% Change the configuration below to use other image or voxel resolutions.

clear;
clc;

scriptDirectory = fileparts(mfilename('fullpath'));
addpath(scriptDirectory);

%% Configurable experiment parameters
config.inputSize = 16;
config.voxelResolution = 16;
config.hiddenUnits = 96;
config.trainingExamples = 510;
config.validationExamples = 85;
config.testExamples = 85;
config.batchSize = 32;
config.learningRate = 0.003;
config.maxEpochs = 3000;
config.patience = 30;
config.minImprovement = 1e-4;
config.weightDecay = 1e-5;
config.maxGradientNorm = 10;
config.positiveClassWeight = 6;
config.occupancyThreshold = 0.5;
config.seed = 2026;

%% Generate independent, deterministic shape splits
[trainingInputs, trainingTargets] = generateSyntheticData( ...
    config.trainingExamples, config.inputSize, config.voxelResolution, config.seed + 1);
[validationInputs, validationTargets] = generateSyntheticData( ...
    config.validationExamples, config.inputSize, config.voxelResolution, config.seed + 2);
[testInputs, testTargets] = generateSyntheticData( ...
    config.testExamples, config.inputSize, config.voxelResolution, config.seed + 3);

network = buildNetwork(config.inputSize, config.voxelResolution, ...
    config.hiddenUnits, mean(trainingTargets(:)), config.seed);
network.occupancyThreshold = config.occupancyThreshold;
trainingOptions.learningRate = config.learningRate;
trainingOptions.maxEpochs = config.maxEpochs;
trainingOptions.batchSize = config.batchSize;
trainingOptions.patience = config.patience;
trainingOptions.minImprovement = config.minImprovement;
trainingOptions.weightDecay = config.weightDecay;
trainingOptions.maxGradientNorm = config.maxGradientNorm;
trainingOptions.positiveClassWeight = config.positiveClassWeight;
trainingOptions.seed = config.seed + 4;

fprintf('Image-to-voxel MLP: %d x %d image -> %d x %d x %d voxels\n', ...
    config.inputSize, config.inputSize, config.voxelResolution, ...
    config.voxelResolution, config.voxelResolution);
fprintf('Synthetic examples: %d training, %d validation, %d test\n', ...
    config.trainingExamples, config.validationExamples, config.testExamples);
network = trainNetwork(network, trainingInputs, trainingTargets, ...
    validationInputs, validationTargets, trainingOptions);
[network.occupancyThreshold, validationVoxelIoU] = ...
    selectOccupancyThreshold(network, validationInputs, validationTargets);
[metrics, testProbabilities] = evaluateNetwork(network, testInputs, testTargets);

fprintf('Epochs: %d / %d; best validation epoch: %d; stopped early: %s\n', ...
    network.epochs, config.maxEpochs, network.bestEpoch, ...
    char({'no', 'yes'}(network.stoppedEarly + 1)));
fprintf('Training BCE: %.6f; best validation BCE: %.6f; test BCE: %.6f\n', ...
    network.trainingLoss, network.validationLoss, metrics.binaryCrossEntropy);
fprintf('Validation-selected occupancy threshold: %.3f; validation voxel IoU: %.4f\n', ...
    network.occupancyThreshold, validationVoxelIoU);
fprintf('Test voxel IoU: %.4f; silhouette IoU: %.4f; voxel accuracy: %.4f\n', ...
    metrics.voxelIoU, metrics.silhouetteIoU, metrics.voxelAccuracy);

%% Export a held-out grayscale input and its predicted mesh
sampleImage = reshape(testInputs(1, :), config.inputSize, config.inputSize);
predictedVolume = applyNetwork(network, sampleImage);
targetVolume = reshape(testTargets(1, :), config.voxelResolution, ...
    config.voxelResolution, config.voxelResolution);
predictedMeshPath = fullfile(scriptDirectory, 'prediction.obj');
targetMeshPath = fullfile(scriptDirectory, 'target.obj');
exportVoxelMesh(predictedVolume, predictedMeshPath, config.occupancyThreshold);
exportVoxelMesh(targetVolume, targetMeshPath, 0.5);
inputImagePath = fullfile(scriptDirectory, 'test-input.png');
imwrite(sampleImage, inputImagePath);
targetSilhouettePath = fullfile(scriptDirectory, 'target-silhouette.png');
predictedSilhouettePath = fullfile(scriptDirectory, 'predicted-silhouette.png');
voxelComparisonPath = fullfile(scriptDirectory, 'voxel-comparison.png');
imwrite(any(targetVolume, 3), targetSilhouettePath);
imwrite(any(predictedVolume >= config.occupancyThreshold, 3), predictedSilhouettePath);
renderVoxelComparisonFigure(targetVolume, predictedVolume, voxelComparisonPath, ...
    config.occupancyThreshold);
fprintf(['Example input: %s\nTarget silhouette: %s\nPredicted silhouette: %s\n' ...
    'Isometric target/prediction comparison: %s\n' ...
    'Predicted mesh: %s\nTarget mesh: %s\n'], inputImagePath, ...
    targetSilhouettePath, predictedSilhouettePath, voxelComparisonPath, ...
    predictedMeshPath, targetMeshPath);

trainingHistory = struct('epoch', network.epochHistory, ...
    'trainingBCE', network.trainingLossHistory, ...
    'validationBCE', network.validationLossHistory);
modelPath = fullfile(scriptDirectory, 'model.mat');
save(modelPath, 'network', 'config', 'metrics', 'trainingHistory', ...
    'validationVoxelIoU', 'testInputs', 'testTargets', 'testProbabilities', '-mat7-binary');
fprintf('Saved trained model and held-out metrics to %s\n', modelPath);