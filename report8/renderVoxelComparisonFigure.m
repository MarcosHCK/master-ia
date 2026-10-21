function renderVoxelComparisonFigure(targetVolume, predictedVolume, outputPath, threshold)
%RENDERVOXELCOMPARISONFIGURE Render target and predicted solids isometrically.

if nargin < 4 || isempty(threshold)
    threshold = 0.5;
end
if ~isnumeric(targetVolume) || ~isreal(targetVolume) || ndims(targetVolume) ~= 3 || ...
        any(~isfinite(targetVolume(:))) || ...
        ~isnumeric(predictedVolume) || ~isreal(predictedVolume) || ...
        ndims(predictedVolume) ~= 3 || any(~isfinite(predictedVolume(:))) || ...
        ~isequal(size(targetVolume), size(predictedVolume))
    error('Target and predicted volumes must be finite, real, matching 3-D arrays.');
end
if ~ischar(outputPath) || isempty(outputPath)
    error('outputPath must be a non-empty character vector.');
end
if ~isscalar(threshold) || ~isfinite(threshold)
    error('threshold must be a finite scalar.');
end

targetSurface = isosurface(double(targetVolume), 0.5);
predictedSurface = isosurface(double(predictedVolume), threshold);
if isempty(targetSurface.faces) || isempty(predictedSurface.faces)
    error('Both volumes must contain a surface at their respective thresholds.');
end

figureHandle = figure('visible', 'off', 'color', 'white', ...
    'units', 'pixels', 'position', [100, 100, 1120, 580]);
closeFigure = onCleanup(@() close(figureHandle));

% Keep a 19-pixel logical gap between the equal-sized panels. At 150 DPI,
% this renders to approximately 30 pixels in the exported PNG.
axesHandle = axes('parent', figureHandle, 'units', 'pixels', ...
    'position', [10, 25, 540, 530]);
drawSurface(axesHandle, targetSurface, size(targetVolume, 1), ...
    [0.20, 0.48, 0.76], 'Target model');

axesHandle = axes('parent', figureHandle, 'units', 'pixels', ...
    'position', [569, 25, 540, 530]);
drawSurface(axesHandle, predictedSurface, size(predictedVolume, 1), ...
    [0.91, 0.43, 0.20], 'Predicted model');

set(figureHandle, 'paperpositionmode', 'auto');
print(figureHandle, outputPath, '-dpng', '-r150');
clear closeFigure;
end

function drawSurface(axesHandle, surface, resolution, faceColor, plotTitle)
patch('parent', axesHandle, 'vertices', surface.vertices, ...
    'faces', surface.faces, 'facecolor', faceColor, ...
    'edgecolor', 'none', 'facelighting', 'gouraud');
axis(axesHandle, [1, resolution, 1, resolution, 1, resolution]);
axis(axesHandle, 'equal');
axis(axesHandle, 'vis3d');
axis(axesHandle, 'off');
view(axesHandle, 45, 35.264);
title(axesHandle, plotTitle, 'fontsize', 16, 'fontweight', 'bold');
camlight(axesHandle, 'headlight');
lighting(axesHandle, 'gouraud');
material(axesHandle, 'dull');
end