function artifactPaths = exportHeldOutArtifacts(network, dataset, outputDirectory, testProbabilities)
%EXPORTHeldOutArtifacts Export held-out depthmaps, silhouettes, and volume meshes.

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
requiredDatasetFields = {'testInputs', 'testTargets'};
if ~isstruct(dataset) || ~all(isfield(dataset, requiredDatasetFields))
    error('dataset must contain testInputs and testTargets.');
end
resolution = network.voxelResolution;
if size(dataset.testInputs, 2) ~= network.inputSize ^ 2 || ...
        size(dataset.testTargets, 1) ~= size(dataset.testInputs, 1) || ...
        size(dataset.testTargets, 2) ~= resolution ^ 3 || isempty(dataset.testInputs)
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

exampleIndex = 1;
testImage = reshape(dataset.testInputs(exampleIndex, :), ...
    network.inputSize, network.inputSize);
targetVolume = reshape(dataset.testTargets(exampleIndex, :), ...
    resolution, resolution, resolution);
predictedProbabilityVolume = reshape(testProbabilities(exampleIndex, :), ...
    resolution, resolution, resolution);
predictedVolume = predictedProbabilityVolume >= network.occupancyThreshold;

artifactPaths.inputDepthmap = fullfile(outputDirectory, 'test-input.png');
artifactPaths.targetSilhouette = fullfile(outputDirectory, 'target-silhouette.png');
artifactPaths.predictedSilhouette = fullfile(outputDirectory, 'predicted-silhouette.png');
artifactPaths.predictedMesh = fullfile(outputDirectory, 'prediction.obj');
artifactPaths.targetMesh = fullfile(outputDirectory, 'target.obj');
artifactPaths.exampleIndex = exampleIndex;
imwrite(testImage, artifactPaths.inputDepthmap);
imwrite(any(targetVolume, 3), artifactPaths.targetSilhouette);
imwrite(any(predictedVolume, 3), artifactPaths.predictedSilhouette);
exportVoxelMesh(predictedProbabilityVolume, artifactPaths.predictedMesh, ...
    network.occupancyThreshold);
exportVoxelMesh(targetVolume, artifactPaths.targetMesh, 0.5);
fprintf('Saved held-out depthmaps, silhouettes, and volume meshes to %s\n', outputDirectory);
end