function result = inferImage(weightsPath, imagePath, outputDirectory)
%INFERIMAGE Load saved weights and reconstruct one grayscale image as voxels.
% Example: inferImage('trained-network.mat', 'input.png')

scriptDirectory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(weightsPath)
    weightsPath = fullfile(scriptDirectory, 'trained-network.mat');
end
if nargin < 2 || isempty(imagePath)
    error('Pass the path to a grayscale or RGB image as imagePath.');
end
if nargin < 3 || isempty(outputDirectory)
    outputDirectory = fileparts(imagePath);
    if isempty(outputDirectory)
        outputDirectory = pwd;
    end
end
if ~ischar(weightsPath) || isempty(weightsPath) || ...
        ~ischar(imagePath) || isempty(imagePath) || ...
        ~ischar(outputDirectory) || isempty(outputDirectory)
    error('weightsPath, imagePath, and outputDirectory must be non-empty character vectors.');
end
if exist(weightsPath, 'file') ~= 2
    error('Weights file does not exist: %s. Run trainModel first.', weightsPath);
end
if exist(imagePath, 'file') ~= 2
    error('Input image does not exist: %s.', imagePath);
end

savedModel = load(weightsPath);
if ~isfield(savedModel, 'network') || ~isfield(savedModel, 'inferenceConfig')
    error('Weights file must contain network and inferenceConfig.');
end
network = savedModel.network;
inferenceConfig = savedModel.inferenceConfig;
if inferenceConfig.inputSize ~= network.inputSize || ...
        inferenceConfig.voxelResolution ~= network.voxelResolution || ...
        inferenceConfig.occupancyThreshold ~= network.occupancyThreshold
    error('Saved inference metadata does not match the network parameters.');
end

sourceImage = readNormalizedGrayscale(imagePath);
inputImage = resizeGrayscaleImage(sourceImage, inferenceConfig.inputSize, ...
    inferenceConfig.inputSize);
probabilityVolume = applyNetwork(network, inputImage);
occupancyVolume = probabilityVolume >= inferenceConfig.occupancyThreshold;
if ~any(occupancyVolume(:))
    error(['The model predicted an empty volume at threshold %.3f. ' ...
        'No OBJ surface can be extracted.'], inferenceConfig.occupancyThreshold);
end

if exist(outputDirectory, 'dir') ~= 7
    [created, message] = mkdir(outputDirectory);
    if ~created
        error('Could not create inference output directory: %s', message);
    end
end
[~, baseName] = fileparts(imagePath);
if isempty(baseName)
    baseName = 'image';
end
result.inputImage = inputImage;
result.probabilityVolume = probabilityVolume;
result.occupancyVolume = occupancyVolume;
result.occupancyThreshold = inferenceConfig.occupancyThreshold;
result.weightsPath = weightsPath;
result.inputPath = imagePath;
result.volumePath = fullfile(outputDirectory, [baseName, '-prediction.mat']);
result.meshPath = fullfile(outputDirectory, [baseName, '-prediction.obj']);
result.silhouettePath = fullfile(outputDirectory, [baseName, '-prediction-silhouette.png']);

exportVoxelMesh(probabilityVolume, result.meshPath, result.occupancyThreshold);
imwrite(any(occupancyVolume, 3), result.silhouettePath);
save(result.volumePath, 'inputImage', 'probabilityVolume', 'occupancyVolume', ...
    'inferenceConfig', 'weightsPath', 'imagePath', '-mat7-binary');
fprintf('Loaded weights: %s\n', weightsPath);
fprintf('Resized grayscale input: %d x %d\n', ...
    inferenceConfig.inputSize, inferenceConfig.inputSize);
fprintf('Saved predicted volume: %s\n', result.volumePath);
fprintf('Saved silhouette: %s\n', result.silhouettePath);
fprintf('Saved 3D mesh: %s\n', result.meshPath);
end

function grayscaleImage = readNormalizedGrayscale(imagePath)
rawImage = imread(imagePath);
if isempty(rawImage) || any(~isfinite(double(rawImage(:))))
    error('The input image must contain finite pixel values.');
end
if islogical(rawImage)
    normalizedImage = double(rawImage);
elseif isinteger(rawImage)
    normalizedImage = double(rawImage) / double(intmax(class(rawImage)));
elseif isfloat(rawImage)
    normalizedImage = double(rawImage);
    if any(normalizedImage(:) < 0) || any(normalizedImage(:) > 1)
        error('Floating-point input pixels must already be in [0,1].');
    end
else
    error('Unsupported image pixel data type.');
end

if ismatrix(normalizedImage)
    grayscaleImage = normalizedImage;
elseif ndims(normalizedImage) == 3 && size(normalizedImage, 3) >= 3
    red = normalizedImage(:, :, 1);
    green = normalizedImage(:, :, 2);
    blue = normalizedImage(:, :, 3);
    grayscaleImage = 0.298936 * red + 0.587043 * green + 0.114021 * blue;
else
    error('The input must be a grayscale image or an RGB image.');
end
end