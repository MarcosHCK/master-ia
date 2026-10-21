% TEST_REPORT8 Small deterministic smoke test for the report8 Octave code.

scriptDirectory = fileparts(mfilename('fullpath'));
addpath(scriptDirectory);

[inputs, targets] = generateSyntheticData(6, 7, 8, 42);
assert(isequal(size(inputs), [6, 49]));
assert(isequal(size(targets), [6, 512]));
assert(all(isfinite(inputs(:))) && all(inputs(:) >= 0) && all(inputs(:) <= 1));
assert(all((targets(:) == 0) | (targets(:) == 1)));

network = buildNetwork(7, 8, 10, mean(targets(:)), 11);
options.learningRate = 0.001;
options.maxEpochs = 3;
options.batchSize = 3;
options.patience = 3;
options.seed = 12;
network = trainNetwork(network, inputs(1:4, :), targets(1:4, :), ...
    inputs(5:6, :), targets(5:6, :), options);
assert(network.epochs >= 1 && network.epochs <= options.maxEpochs);
assert(isfinite(network.validationLoss));

singlePrediction = applyNetwork(network, reshape(inputs(5, :), 7, 7));
batchPredictions = applyNetwork(network, inputs(5:6, :));
assert(isequal(size(singlePrediction), [8, 8, 8]));
assert(isequal(size(batchPredictions), [2, 512]));
assert(all(isfinite(singlePrediction(:))) && ...
    all(singlePrediction(:) >= 0) && all(singlePrediction(:) <= 1));

[metrics, ~] = evaluateNetwork(network, inputs(5:6, :), targets(5:6, :));
assert(isfinite(metrics.binaryCrossEntropy));
assert(metrics.voxelIoU >= 0 && metrics.voxelIoU <= 1);
assert(metrics.silhouetteIoU >= 0 && metrics.silhouetteIoU <= 1);
[threshold, validationIoU] = selectOccupancyThreshold(network, ...
    inputs(5:6, :), targets(5:6, :));
assert(threshold >= 0.01 && threshold <= 0.5);
assert(validationIoU >= 0 && validationIoU <= 1);

cube = zeros(8, 8, 8);
cube(3:6, 3:6, 3:6) = 1;
meshPath = [tempname(), '.obj'];
faces = exportVoxelMesh(cube, meshPath, 0.5);
assert(~isempty(faces) && exist(meshPath, 'file') == 2);
meshInfo = dir(meshPath);
assert(meshInfo.bytes > 0);
delete(meshPath);

fprintf('report8 smoke tests passed (synthetic data, training, inference, metrics, OBJ export).\n');