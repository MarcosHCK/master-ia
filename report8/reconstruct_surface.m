function [heightMap, outputFiles] = reconstruct_surface(imageInput, modelFile, ...
        outputPrefix, heightScale)
%RECONSTRUCT_SURFACE Predict an N-by-N surface from a grayscale image.
%   HEIGHTMAP = RECONSTRUCT_SURFACE(IMAGEFILE) loads trained-network.mat
%   beside this file, reads IMAGEFILE, and returns a height map. IMAGEFILE
%   may also be an N-by-N numeric grayscale matrix. For additional outputs,
%   set MODELFILE and OUTPUTPREFIX; outputs are PREFIX-prediction.mat and
%   PREFIX-prediction.obj. Optional HEIGHTSCALE (default 1) multiplies the
%   centered predicted intensity, so height = HEIGHTSCALE*(intensity - 0.5).
%   The scale must be a finite, nonnegative scalar. OBJ is a single open
%   surface mesh, not a volume.

scriptDirectory = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(imageInput)
    error('Provide an image path or an N-by-N grayscale image matrix.');
end
if nargin < 2 || isempty(modelFile)
    modelFile = fullfile(scriptDirectory, 'trained-network.mat');
end
if nargin < 3 || isempty(outputPrefix)
    outputPrefix = fullfile(scriptDirectory, 'prediction');
end
if nargin < 4 || isempty(heightScale)
    heightScale = 1;
end
if ~ischar(modelFile) || ~ischar(outputPrefix)
    error('modelFile and outputPrefix must be character vectors.');
end
if ~isnumeric(heightScale) || ~isreal(heightScale) || ...
        ~isscalar(heightScale) || ~surface_isfinite(heightScale) || heightScale < 0
    error('heightScale must be a finite, nonnegative real scalar.');
end
if exist(modelFile, 'file') ~= 2
    error('Trained model not found: %s', modelFile);
end
loadedModel = load(modelFile, 'network');
if ~isfield(loadedModel, 'network')
    error('Model file must contain a variable named network.');
end
network = loadedModel.network;

if ischar(imageInput)
    if exist(imageInput, 'file') ~= 2
        error('Input image not found: %s', imageInput);
    end
    inputImage = imread(imageInput);
else
    inputImage = imageInput;
end
inputImage = normalizeGrayscaleImage(inputImage);
if ~isequal(size(inputImage), [network.N, network.N])
    error('Image must be exactly %d-by-%d; received %d-by-%d.', ...
        network.N, network.N, size(inputImage, 1), size(inputImage, 2));
end

predictedIntensity = surface_cnn_predict(network, inputImage);
heightMap = heightScale * (predictedIntensity - 0.5);
outputFiles.heightMap = [outputPrefix, '-prediction.mat'];
outputFiles.mesh = [outputPrefix, '-prediction.obj'];
outputDirectory = fileparts(outputPrefix);
if ~isempty(outputDirectory) && exist(outputDirectory, 'dir') ~= 7
    error('The output directory does not exist: %s', outputDirectory);
end
save(outputFiles.heightMap, 'heightMap', 'predictedIntensity', ...
    'inputImage', 'heightScale', '-v7');
writeSurfaceObj(outputFiles.mesh, heightMap);
fprintf('Saved %d-by-%d height map to %s and surface mesh to %s.\n', ...
    network.N, network.N, outputFiles.heightMap, outputFiles.mesh);
end

function grayscale = normalizeGrayscaleImage(image)
if ~isnumeric(image) && ~islogical(image)
    error('Image must be a numeric grayscale or RGB image.');
end
if ~isreal(image) || isempty(image) || ...
        any(~surface_isfinite(double(image(:))))
    error('Image must contain finite real pixel values.');
end
if ndims(image) == 3
    if size(image, 3) < 3
        error('A color input image must have at least three channels.');
    end
    channels = double(image(:, :, 1:3));
    grayscale = 0.298936 * channels(:, :, 1) + ...
        0.587043 * channels(:, :, 2) + 0.114021 * channels(:, :, 3);
elseif ismatrix(image)
    grayscale = double(image);
else
    error('Image must be a two-dimensional grayscale or three-channel RGB image.');
end

if isinteger(image)
    grayscale = grayscale / double(intmax(class(image)));
elseif islogical(image)
    grayscale = double(grayscale);
elseif any(grayscale(:) < 0) || any(grayscale(:) > 1)
    error('Floating-point image intensities must already be in [0,1].');
end
grayscale = min(1, max(0, grayscale));
end

function writeSurfaceObj(filePath, heights)
[height, width] = size(heights);
fileId = fopen(filePath, 'w');
if fileId < 0
    error('Unable to open OBJ output file: %s', filePath);
end
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Predicted open height-field surface; one mesh, no volume.\n');
for row = 1:height
    for column = 1:width
        fprintf(fileId, 'v %.9g %.9g %.9g\n', ...
            column - 1, row - 1, heights(row, column));
    end
end
for row = 1:(height - 1)
    for column = 1:(width - 1)
        topLeft = (row - 1) * width + column;
        topRight = topLeft + 1;
        bottomLeft = row * width + column;
        bottomRight = bottomLeft + 1;
        % Two triangles per grid cell, with upward-facing vertex winding.
        fprintf(fileId, 'f %d %d %d\n', topLeft, topRight, bottomLeft);
        fprintf(fileId, 'f %d %d %d\n', topRight, bottomRight, bottomLeft);
    end
end
clear cleanup;
end