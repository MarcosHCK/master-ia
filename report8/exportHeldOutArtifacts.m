function artifactPaths = exportHeldOutArtifacts(network, dataset, outputDirectory, testHeights)
%EXPORTHeldOutArtifacts Export held-out depthmaps, height maps, and surface OBJ files.

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
if size(dataset.testInputs, 2) ~= network.inputSize ^ 2 || ...
        size(dataset.testTargets, 1) ~= size(dataset.testInputs, 1) || ...
        size(dataset.testTargets, 2) ~= network.surfaceResolution ^ 2 || ...
        isempty(dataset.testInputs)
    error('The held-out split dimensions do not match the saved network.');
end
if nargin < 4 || isempty(testHeights)
    testHeights = applyNetwork(network, dataset.testInputs);
end
if ~isequal(size(testHeights), size(dataset.testTargets)) || ...
        any(~isfinite(testHeights(:))) || ...
        any(testHeights(:) < 0) || any(testHeights(:) > 1)
    error('testHeights must match the held-out targets and lie in [0,1].');
end

exampleIndex = 1;
testImage = reshape(dataset.testInputs(exampleIndex, :), ...
    network.inputSize, network.inputSize);
targetHeightMap = reshape(dataset.testTargets(exampleIndex, :), ...
    network.surfaceResolution, network.surfaceResolution);
predictedHeightMap = reshape(testHeights(exampleIndex, :), ...
    network.surfaceResolution, network.surfaceResolution);
artifactPaths.inputDepthmap = fullfile(outputDirectory, 'test-input.png');
artifactPaths.targetHeightmap = fullfile(outputDirectory, 'target-heightmap.png');
artifactPaths.predictedHeightmap = fullfile(outputDirectory, 'predicted-heightmap.png');
artifactPaths.predictedSurface = fullfile(outputDirectory, 'prediction.obj');
artifactPaths.targetSurface = fullfile(outputDirectory, 'target.obj');
artifactPaths.exampleIndex = exampleIndex;

imwrite(testImage, artifactPaths.inputDepthmap);
imwrite(targetHeightMap, artifactPaths.targetHeightmap);
imwrite(predictedHeightMap, artifactPaths.predictedHeightmap);
exportSurfaceMesh(predictedHeightMap, artifactPaths.predictedSurface);
exportSurfaceMesh(targetHeightMap, artifactPaths.targetSurface);
fprintf('Saved held-out depthmaps, height maps, and surfaces to %s\n', outputDirectory);
end