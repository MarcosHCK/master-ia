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
options.maxEpochs = 20;
options.batchSize = 3;
options.patience = 30;
options.seed = 12;
trainingOutput = evalc('network = trainNetwork(network, inputs(1:4, :), targets(1:4, :), inputs(5:6, :), targets(5:6, :), options);');
assert(network.epochs >= 1 && network.epochs <= options.maxEpochs);
assert(isfinite(network.validationLoss));
progressLines = regexp(trainingOutput, 'Training progress: [^\n]+', 'match');
assert(numel(progressLines) == 20);
for progressPercent = 5:5:100
    expectedProgressMessage = sprintf('Training progress: %d%% (epoch %d/20;', ...
        progressPercent, progressPercent / 5);
    assert(~isempty(strfind(trainingOutput, expectedProgressMessage)));
end

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

figurePath = [tempname(), '.png'];
renderVoxelComparisonFigure(cube, cube, figurePath, 0.5);
assert(exist(figurePath, 'file') == 2);
figureInfo = dir(figurePath);
assert(figureInfo.bytes > 0);
delete(figurePath);

temporaryDirectory = tempname();
[created, message] = mkdir(temporaryDirectory);
assert(created, message);
datasetPath = fullfile(temporaryDirectory, 'test-examples.mat');
weightsPath = fullfile(temporaryDirectory, 'test-network.mat');
dataConfig.inputSize = 7;
dataConfig.voxelResolution = 8;
dataConfig.trainingExamples = 4;
dataConfig.validationExamples = 2;
dataConfig.testExamples = 2;
dataConfig.seed = 43;
dataset = generateExamples(datasetPath, dataConfig);
assert(exist(datasetPath, 'file') == 2);
assert(isequal(size(dataset.trainingInputs), [4, 49]));

trainingOptions.hiddenUnits = 10;
trainingOptions.learningRate = 0.001;
trainingOptions.maxEpochs = 5;
trainingOptions.batchSize = 2;
trainingOptions.patience = 10;
trainingOptions.seed = 44;
trainingOptions.exportEvaluationArtifacts = false;
[savedNetwork, trainingSummary] = trainModel(datasetPath, weightsPath, trainingOptions);
assert(exist(weightsPath, 'file') == 2);
assert(trainingSummary.epochs >= 1 && trainingSummary.epochs <= trainingOptions.maxEpochs);
savedModel = load(weightsPath);
assert(isfield(savedModel, 'network') && isfield(savedModel, 'inferenceConfig'));
assert(~isfield(savedModel, 'trainingInputs') && ~isfield(savedModel, 'testInputs'));
assert(~isfield(savedModel.network, 'trainingLossHistory'));
assert(isequal(size(savedNetwork.W1), size(savedModel.network.W1)));

inferenceInputPath = fullfile(temporaryDirectory, 'inference-input.png');
imwrite(reshape(dataset.testInputs(1, :), 7, 7), inferenceInputPath);
inferenceOutputDirectory = fullfile(temporaryDirectory, 'inference-output');
inferenceResult = inferImage(weightsPath, inferenceInputPath, inferenceOutputDirectory);
assert(isequal(size(inferenceResult.probabilityVolume), [8, 8, 8]));
assert(exist(inferenceResult.volumePath, 'file') == 2);
assert(exist(inferenceResult.meshPath, 'file') == 2);
assert(exist(inferenceResult.silhouettePath, 'file') == 2);

inferenceFiles = dir(inferenceOutputDirectory);
for fileIndex = 1:numel(inferenceFiles)
    if ~inferenceFiles(fileIndex).isdir
        delete(fullfile(inferenceOutputDirectory, inferenceFiles(fileIndex).name));
    end
end
rmdir(inferenceOutputDirectory);
delete(inferenceInputPath);
delete(datasetPath);
delete(weightsPath);
rmdir(temporaryDirectory);

% Smoke-test the three standalone command scripts with small temporary data.
pipelineDirectory = tempname();
[created, message] = mkdir(pipelineDirectory);
assert(created, message);
generateScript = fullfile(scriptDirectory, 'generate_examples.m');
trainScript = fullfile(scriptDirectory, 'train_network.m');
inferenceScript = fullfile(scriptDirectory, 'infer_image.m');
scriptContents = fileread(generateScript);
scriptContents = strrep(scriptContents, ...
    "fullfile(scriptDirectory, 'examples.mat')", ...
    sprintf("fullfile(scriptDirectory, '%s')", 'pipeline-test-examples.mat'));
scriptContents = strrep(scriptContents, 'datasetConfig.inputSize = 16;', ...
    'datasetConfig.inputSize = 7;');
scriptContents = strrep(scriptContents, 'datasetConfig.voxelResolution = 16;', ...
    'datasetConfig.voxelResolution = 8;');
scriptContents = strrep(scriptContents, 'datasetConfig.trainingExamples = 510;', ...
    'datasetConfig.trainingExamples = 4;');
scriptContents = strrep(scriptContents, 'datasetConfig.validationExamples = 85;', ...
    'datasetConfig.validationExamples = 2;');
scriptContents = strrep(scriptContents, 'datasetConfig.testExamples = 85;', ...
    'datasetConfig.testExamples = 2;');
fileId = fopen(fullfile(pipelineDirectory, 'generate_examples.m'), 'w');
assert(fileId >= 0);
fprintf(fileId, '%s', scriptContents);
fclose(fileId);

scriptContents = fileread(trainScript);
scriptContents = strrep(scriptContents, ...
    "fullfile(scriptDirectory, 'examples.mat')", ...
    sprintf("fullfile(scriptDirectory, '%s')", 'pipeline-test-examples.mat'));
scriptContents = strrep(scriptContents, ...
    "fullfile(scriptDirectory, 'trained-network.mat')", ...
    sprintf("fullfile(scriptDirectory, '%s')", 'pipeline-test-network.mat'));
scriptContents = strrep(scriptContents, 'trainingOptions.hiddenUnits = 96;', ...
    'trainingOptions.hiddenUnits = 10;');
scriptContents = strrep(scriptContents, 'trainingOptions.maxEpochs = 3000;', ...
    'trainingOptions.maxEpochs = 5;');
scriptContents = strrep(scriptContents, 'trainingOptions.batchSize = 32;', ...
    'trainingOptions.batchSize = 2;');
scriptContents = strrep(scriptContents, 'trainingOptions.patience = 30;', ...
    'trainingOptions.patience = 10;');
scriptContents = strrep(scriptContents, 'trainingOptions.exportEvaluationArtifacts = true;', ...
    'trainingOptions.exportEvaluationArtifacts = false;');
fileId = fopen(fullfile(pipelineDirectory, 'train_network.m'), 'w');
assert(fileId >= 0);
fprintf(fileId, '%s', scriptContents);
fclose(fileId);

pipelineInputPath = fullfile(pipelineDirectory, 'input.png');
imwrite(reshape(dataset.testInputs(1, :), 7, 7), pipelineInputPath);
scriptContents = fileread(inferenceScript);
scriptContents = strrep(scriptContents, ...
    "fullfile(scriptDirectory, 'trained-network.mat')", ...
    sprintf("fullfile(scriptDirectory, '%s')", 'pipeline-test-network.mat'));
scriptContents = strrep(scriptContents, ...
    "fullfile(scriptDirectory, 'input.png')", ...
    sprintf("fullfile(scriptDirectory, '%s')", 'input.png'));
scriptContents = strrep(scriptContents, ...
    'outputDirectory = scriptDirectory;', ...
    sprintf("outputDirectory = fullfile(scriptDirectory, '%s');", 'pipeline-test-output'));
fileId = fopen(fullfile(pipelineDirectory, 'infer_image.m'), 'w');
assert(fileId >= 0);
fprintf(fileId, '%s', scriptContents);
fclose(fileId);

previousDirectory = pwd();
restoreDirectory = onCleanup(@() cd(previousDirectory));
addpath(pipelineDirectory);
cd(pipelineDirectory);
evalc('run(''generate_examples.m'');');
evalc('run(''train_network.m'');');
evalc('run(''infer_image.m'');');
assert(exist(fullfile(pipelineDirectory, 'pipeline-test-examples.mat'), 'file') == 2);
assert(exist(fullfile(pipelineDirectory, 'pipeline-test-network.mat'), 'file') == 2);
assert(exist(fullfile(pipelineDirectory, 'pipeline-test-output', ...
    'input-prediction.obj'), 'file') == 2);
clear restoreDirectory;
rmpath(pipelineDirectory);
delete(fullfile(pipelineDirectory, 'pipeline-test-examples.mat'));
delete(fullfile(pipelineDirectory, 'pipeline-test-network.mat'));
delete(fullfile(pipelineDirectory, 'input.png'));
rmdir(pipelineDirectory, 's');

fprintf(['report8 smoke tests passed (synthetic data, training, inference, ' ...
    'metrics, mesh/figure export, save/load, and three-script workflow).\n']);