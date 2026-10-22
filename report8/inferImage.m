function result = inferImage(weightsPath, imagePath, outputDirectory)
%INFERIMAGE Reconstruct a 3-D height-field surface from a top-down depthmap.
% Example: inferImage('trained-network.mat', 'depthmap.png')

scriptDirectory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(weightsPath)
    weightsPath = fullfile(scriptDirectory, 'trained-network.mat');
end
if nargin < 2 || isempty(imagePath)
    error('Pass the path to a grayscale or RGB depthmap image as imagePath.');
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
if ~isfield(network, 'surfaceResolution') || ...
        ~isfield(inferenceConfig, 'surfaceResolution')
    error(['This checkpoint uses the previous voxel-occupancy model. ' ...
        'Regenerate the dataset and retrain before running surface inference.']);
end
if ~isfield(inferenceConfig, 'inputSize') || ...
        inferenceConfig.inputSize ~= network.inputSize || ...
        inferenceConfig.surfaceResolution ~= network.surfaceResolution
    error('Saved inference metadata does not match the network parameters.');
end

sourceImage = readNormalizedGrayscale(imagePath);
inputImage = resizeGrayscaleImage(sourceImage, inferenceConfig.inputSize, ...
    inferenceConfig.inputSize);
heightMap = applyNetwork(network, inputImage);

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
result.heightMap = heightMap;
result.surfacePath = fullfile(outputDirectory, [baseName, '-prediction.obj']);
result.heightMapPath = fullfile(outputDirectory, [baseName, '-prediction-heightmap.png']);
result.dataPath = fullfile(outputDirectory, [baseName, '-prediction.mat']);
result.meshPath = result.surfacePath;
result.weightsPath = weightsPath;
result.inputPath = imagePath;

exportSurfaceMesh(heightMap, result.surfacePath);
imwrite(heightMap, result.heightMapPath);
save(result.dataPath, 'inputImage', 'heightMap', 'inferenceConfig', ...
    'weightsPath', 'imagePath', '-mat7-binary');
fprintf('Loaded weights: %s\n', weightsPath);
fprintf('Resized input depthmap: %d x %d\n', ...
    inferenceConfig.inputSize, inferenceConfig.inputSize);
fprintf('Saved predicted height map: %s\n', result.heightMapPath);
fprintf('Saved reconstructed 3D surface: %s\n', result.surfacePath);
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