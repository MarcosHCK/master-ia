function artifactPaths = exportHeldOutArtifacts(network, dataset, outputDirectory, testProbabilities)
%EXPORTHeldOutArtifacts Export held-out images, silhouettes, 3-D figure, and meshes.
% This can be run after training to regenerate report outputs from a saved model.

if nargin < 3 || isempty(outputDirectory)
    outputDirectory = pwd;
end
if ~ischar(outputDirectory) || isempty(outputDirectory)
    error('outputDirectory must be a non-empty character vector.');
end
if exist(outputDirectory, 'dir') ~= 7
    [created, message] = mkdir(outputDirectory);
    if ~created
        error('Could not create output directory: %s', message);
    end
end
requiredDatasetFields = {'testInputs', 'testTargets', 'config'};
if ~isstruct(dataset) || ~all(isfield(dataset, requiredDatasetFields))
    error('dataset must contain testInputs, testTargets, and config.');
end
requiredNetworkFields = {'inputSize', 'voxelResolution', 'occupancyThreshold'};
if ~isstruct(network) || ~all(isfield(network, requiredNetworkFields))
    error('network must contain input/output dimensions and occupancyThreshold.');
end
if size(dataset.testInputs, 2) ~= network.inputSize ^ 2 || ...
        size(dataset.testTargets, 1) ~= size(dataset.testInputs, 1) || ...
        size(dataset.testTargets, 2) ~= network.voxelResolution ^ 3 || ...
        isempty(dataset.testInputs)
    error('The held-out split dimensions do not match the saved network.');
end
if nargin < 4 || isempty(testProbabilities)
    testProbabilities = applyNetwork(network, dataset.testInputs);
end
if ~isequal(size(testProbabilities), size(dataset.testTargets)) || ...
        any(~isfinite(testProbabilities(:))) || ...
        any(testProbabilities(:) < 0) || any(testProbabilities(:) > 1)
    error('testProbabilities must match the held-out targets and lie in [0,1].');
end

inferenceConfig = struct('inputSize', network.inputSize, ...
    'voxelResolution', network.voxelResolution);
exampleIndex = 1;
testImage = reshape(dataset.testInputs(exampleIndex, :), ...
    inferenceConfig.inputSize, inferenceConfig.inputSize);
targetVolume = reshape(dataset.testTargets(exampleIndex, :), ...
    inferenceConfig.voxelResolution, inferenceConfig.voxelResolution, ...
    inferenceConfig.voxelResolution);
predictedProbabilityVolume = reshape(testProbabilities(exampleIndex, :), ...
    inferenceConfig.voxelResolution, inferenceConfig.voxelResolution, ...
    inferenceConfig.voxelResolution);
predictedVolume = predictedProbabilityVolume >= network.occupancyThreshold;

artifactPaths.inputImage = fullfile(outputDirectory, 'test-input.png');
artifactPaths.targetSilhouette = fullfile(outputDirectory, 'target-silhouette.png');
artifactPaths.predictedSilhouette = fullfile(outputDirectory, 'predicted-silhouette.png');
artifactPaths.voxelComparison = fullfile(outputDirectory, 'voxel-comparison.png');
artifactPaths.predictedMesh = fullfile(outputDirectory, 'prediction.obj');
artifactPaths.targetMesh = fullfile(outputDirectory, 'target.obj');
artifactPaths.exampleIndex = exampleIndex;

imwrite(testImage, artifactPaths.inputImage);
imwrite(any(targetVolume, 3), artifactPaths.targetSilhouette);
imwrite(any(predictedVolume, 3), artifactPaths.predictedSilhouette);
renderVoxelComparisonFigure(targetVolume, predictedProbabilityVolume, ...
    artifactPaths.voxelComparison, network.occupancyThreshold);
exportVoxelMesh(predictedProbabilityVolume, artifactPaths.predictedMesh, ...
    network.occupancyThreshold);
exportVoxelMesh(targetVolume, artifactPaths.targetMesh, 0.5);
fprintf('Saved held-out comparison images and meshes to %s\n', outputDirectory);
end