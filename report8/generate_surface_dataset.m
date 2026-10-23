function dataset = generate_surface_dataset(outputFile, numExamples, N, noiseStd, seed)
%GENERATE_SURFACE_DATASET Generate smooth synthetic surfaces and image inputs.
%   DATASET = GENERATE_SURFACE_DATASET() creates 320 examples at 64-by-64,
%   adds Gaussian observation noise with standard deviation 0.025, and saves
%   the train/validation/test splits to examples.mat beside this file.
%
%   DATASET = GENERATE_SURFACE_DATASET(FILE, COUNT, N, NOISESTD, SEED)
%   sets the MAT-file path, total example count, square image size, input
%   noise standard deviation (in grayscale units), and reproducible seed.
%   Set NOISESTD to zero for noiseless input/target pairs. Targets always
%   remain clean. Each target height is grayscale intensity minus 0.5.
%
%   DATASET.train.inputImages and DATASET.train.targetHeights have dimensions
%   N-by-N-by-example. Validation and test splits use the same layout.
%   Uses only MATLAB/Octave built-ins.

scriptDirectory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outputFile)
    outputFile = fullfile(scriptDirectory, 'examples.mat');
end
if nargin < 2 || isempty(numExamples)
    numExamples = 320;
end
if nargin < 3 || isempty(N)
    N = 64;
end
if nargin < 4 || isempty(noiseStd)
    noiseStd = 0.025;
end
if nargin < 5 || isempty(seed)
    seed = 1;
end

if ~ischar(outputFile)
    error('outputFile must be a character vector containing a file path.');
end
if ~isnumeric(numExamples) || ~isscalar(numExamples) || ...
        ~surface_isfinite(numExamples) || numExamples < 7 || ...
        numExamples ~= floor(numExamples)
    error('numExamples must be a finite integer greater than or equal to 7.');
end
if ~isnumeric(N) || ~isscalar(N) || ~surface_isfinite(N) || ...
        N < 8 || N ~= floor(N)
    error('N must be a finite integer greater than or equal to 8.');
end
if ~isnumeric(noiseStd) || ~isscalar(noiseStd) || ...
        ~surface_isfinite(noiseStd) || noiseStd < 0
    error('noiseStd must be a finite, nonnegative scalar.');
end
if ~isnumeric(seed) || ~isscalar(seed) || ~surface_isfinite(seed) || ...
        seed < 0 || seed ~= floor(seed)
    error('seed must be a finite, nonnegative integer.');
end

% Preserve the caller's random state while making generated datasets repeatable.
previousRngState = rng;
rng(seed, 'twister');
rngCleanup = onCleanup(@() rng(previousRngState));
grid = linspace(-1, 1, N);
[xGrid, yGrid] = meshgrid(grid, grid);
cleanImages = zeros(N, N, numExamples);
noisyImages = zeros(N, N, numExamples);
targetHeights = zeros(N, N, numExamples);

for example = 1:numExamples
    % A low-frequency wave makes each surface vary smoothly across the image.
    angle = 2 * pi * rand;
    frequencyX = 0.45 + 1.1 * rand;
    frequencyY = 0.45 + 1.1 * rand;
    phase = 2 * pi * rand;
    surface = 0.35 * sin(frequencyX * pi * ...
        (cos(angle) * xGrid + sin(angle) * yGrid) + phase);

    % Overlapping Gaussian hills and depressions create soft peaks and holes.
    numFeatures = 3 + floor(5 * rand);
    for feature = 1:numFeatures
        centerX = -0.78 + 1.56 * rand;
        centerY = -0.78 + 1.56 * rand;
        radiusX = 0.12 + 0.42 * rand;
        radiusY = 0.12 + 0.42 * rand;
        amplitude = (0.25 + 0.85 * rand) * (2 * (rand >= 0.5) - 1);
        gaussian = exp(-0.5 * (((xGrid - centerX) / radiusX).^2 + ...
            ((yGrid - centerY) / radiusY).^2));
        surface = surface + amplitude * gaussian;
    end

    % Keep the nominal level at 50% and vary the peak/hole depth per example.
    surface = surface - mean(surface(:));
    maximumDeviation = max(abs(surface(:)));
    if maximumDeviation > 0
        relief = 0.12 + 0.30 * rand;
        cleanImage = 0.5 + relief * surface / maximumDeviation;
    else
        cleanImage = 0.5 * ones(N, N);
    end
    cleanImage = min(1, max(0, cleanImage));
    noisyImage = min(1, max(0, cleanImage + noiseStd * randn(N, N)));

    cleanImages(:, :, example) = cleanImage;
    noisyImages(:, :, example) = noisyImage;
    targetHeights(:, :, example) = cleanImage - 0.5;
end

% Split by entire surfaces, never by pixels from the same surface.
permutation = randperm(numExamples);
numTrain = floor(0.70 * numExamples);
numValidation = floor(0.15 * numExamples);
numTest = numExamples - numTrain - numValidation;
trainIndices = permutation(1:numTrain);
validationIndices = permutation(numTrain + 1:numTrain + numValidation);
testIndices = permutation(numTrain + numValidation + 1:end);

dataset.formatVersion = 1;
dataset.N = N;
dataset.noiseStd = noiseStd;
dataset.seed = seed;
dataset.train.inputImages = noisyImages(:, :, trainIndices);
dataset.train.targetHeights = targetHeights(:, :, trainIndices);
dataset.train.cleanImages = cleanImages(:, :, trainIndices);
dataset.validation.inputImages = noisyImages(:, :, validationIndices);
dataset.validation.targetHeights = targetHeights(:, :, validationIndices);
dataset.validation.cleanImages = cleanImages(:, :, validationIndices);
dataset.test.inputImages = noisyImages(:, :, testIndices);
dataset.test.targetHeights = targetHeights(:, :, testIndices);
dataset.test.cleanImages = cleanImages(:, :, testIndices);

outputDirectory = fileparts(outputFile);
if ~isempty(outputDirectory) && exist(outputDirectory, 'dir') ~= 7
    error('The output directory does not exist: %s', outputDirectory);
end
save(outputFile, 'dataset', '-v7');
clear rngCleanup;

fprintf('Saved %d smooth %d-by-%d surfaces to %s (noiseStd = %.4f).\n', ...
    numExamples, N, N, outputFile, noiseStd);
fprintf('Split: %d training, %d validation, %d test examples.\n', ...
    numTrain, numValidation, numTest);
end