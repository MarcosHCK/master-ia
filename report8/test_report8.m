% TEST_REPORT8 Deterministic smoke test for the report8 Octave workflow.

scriptDirectory = fileparts(mfilename('fullpath'));
addpath(scriptDirectory);

[inputs, targets] = generateSyntheticData(6, 7, 8, 42);
assert(isequal(size(inputs), [6, 49]));
assert(isequal(size(targets), [6, 64]));
assert(all(isfinite(inputs(:))) && all(inputs(:) >= 0) && all(inputs(:) <= 1));
assert(all(isfinite(targets(:))) && all(targets(:) >= 0) && all(targets(:) <= 1));
[repeatInputs, repeatTargets] = generateSyntheticData(6, 7, 8, 42);
assert(isequal(inputs, repeatInputs) && isequal(targets, repeatTargets));

network = buildNetwork(7, 8, 10, mean(targets(:)), 11);
initialTrainingPredictions = applyNetwork(network, inputs(1:4, :));
initialTrainingErrors = initialTrainingPredictions - targets(1:4, :);
initialTrainingLoss = mean(initialTrainingErrors(:) .^ 2);
options.learningRate = 0.01;
options.maxEpochs = 20;
options.batchSize = 3;
options.patience = 30;
options.seed = 12;
trainingOutput = evalc('network = trainNetwork(network, inputs(1:4, :), targets(1:4, :), inputs(5:6, :), targets(5:6, :), options);');
assert(network.epochs >= 1 && network.epochs <= options.maxEpochs);
assert(isfinite(network.validationLoss));
assert(network.trainingLoss < initialTrainingLoss);
progressLines = regexp(trainingOutput, 'Training progress: [^\n]+', 'match');
assert(numel(progressLines) == 20);
for progressPercent = 5:5:100
    expectedProgressMessage = sprintf('Training progress: %d%% (epoch %d/20;', ...
        progressPercent, progressPercent / 5);
    assert(~isempty(strfind(trainingOutput, expectedProgressMessage)));
end

singlePrediction = applyNetwork(network, reshape(inputs(5, :), 7, 7));
batchPredictions = applyNetwork(network, inputs(5:6, :));
assert(isequal(size(singlePrediction), [8, 8]));
assert(isequal(size(batchPredictions), [2, 64]));
assert(all(isfinite(singlePrediction(:))) && ...
    all(singlePrediction(:) >= 0) && all(singlePrediction(:) <= 1));

[metrics, ~] = evaluateNetwork(network, inputs(5:6, :), targets(5:6, :));
assert(isfinite(metrics.meanSquaredError) && metrics.meanSquaredError >= 0);
assert(isfinite(metrics.rootMeanSquaredError) && metrics.rootMeanSquaredError >= 0);
assert(isfinite(metrics.meanAbsoluteError) && metrics.meanAbsoluteError >= 0);

heightMap = reshape(targets(1, :), 8, 8);
meshPath = [tempname(), '.obj'];
faces = exportSurfaceMesh(heightMap, meshPath);
assert(isequal(size(faces), [2 * 7 * 7, 3]));
assert(exist(meshPath, 'file') == 2);
meshInfo = dir(meshPath);
assert(meshInfo.bytes > 0);
delete(meshPath);

temporaryDirectory = tempname();
[created, message] = mkdir(temporaryDirectory);
assert(created, message);
datasetPath = fullfile(temporaryDirectory, 'test-examples.mat');
weightsPath = fullfile(temporaryDirectory, 'test-network.mat');
dataConfig.inputSize = 11;
dataConfig.surfaceResolution = 8;
dataConfig.trainingExamples = 4;
dataConfig.validationExamples = 2;
dataConfig.testExamples = 2;
dataConfig.seed = 43;
dataset = generateExamples(datasetPath, dataConfig);
assert(exist(datasetPath, 'file') == 2);
assert(isequal(size(dataset.trainingInputs), [4, 121]));
assert(isequal(size(dataset.trainingTargets), [4, 64]));

trainingOptions.hiddenUnits = 10;
trainingOptions.learningRate = 0.01;
trainingOptions.maxEpochs = 5;
trainingOptions.batchSize = 2;
trainingOptions.patience = 10;
trainingOptions.seed = 44;
trainingOptions.exportEvaluationArtifacts = true;
[savedNetwork, trainingSummary] = trainModel(datasetPath, weightsPath, trainingOptions);
assert(exist(weightsPath, 'file') == 2);
assert(trainingSummary.epochs >= 1 && trainingSummary.epochs <= trainingOptions.maxEpochs);
assert(isfield(trainingSummary.testMetrics, 'rootMeanSquaredError'));
assert(savedNetwork.surfaceResolution == 8);
assert(savedNetwork.inputSize == 11);
assert(isequal(size(savedNetwork.W1), [121, trainingOptions.hiddenUnits]));

inputPath = fullfile(temporaryDirectory, 'input.png');
imwrite(reshape(dataset.testInputs(1, :), 11, 11), inputPath);
inferenceDirectory = fullfile(temporaryDirectory, 'inference');
result = inferImage(weightsPath, inputPath, inferenceDirectory);
assert(isequal(size(result.inputImage), [11, 11]));
assert(isequal(size(result.heightMap), [8, 8]));
assert(exist(result.surfacePath, 'file') == 2);
assert(exist(result.heightMapPath, 'file') == 2);
assert(exist(result.dataPath, 'file') == 2);
assert(exist(fullfile(fileparts(weightsPath), 'prediction.obj'), 'file') == 2);

rmdir(temporaryDirectory, 's');
fprintf(['report8 smoke tests passed (depthmap data, height regression, ' ...
    'training, inference, metrics, OBJ surface export, and checkpoint workflow).\n']);