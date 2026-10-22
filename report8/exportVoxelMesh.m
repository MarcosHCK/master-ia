function faces = exportVoxelMesh(volume, outputPath, threshold)
%EXPORTVOXELMESH Extract an isosurface and write it as a triangular OBJ mesh.

if nargin < 3 || isempty(threshold)
    threshold = 0.5;
end
if ~isnumeric(volume) || ~isreal(volume) || ndims(volume) ~= 3 || ...
        any(~isfinite(volume(:))) || any(volume(:) < 0) || any(volume(:) > 1)
    error('volume must be a finite, real 3-D occupancy array in [0,1].');
end
if ~ischar(outputPath) || isempty(outputPath)
    error('outputPath must be a non-empty character vector.');
end
if ~isscalar(threshold) || ~isfinite(threshold) || threshold < 0 || threshold > 1
    error('threshold must be a finite scalar in [0,1].');
end
resolution = size(volume, 1);
if size(volume, 2) ~= resolution || size(volume, 3) ~= resolution || resolution < 2
    error('Voxel volumes must be cubic and have a resolution of at least 2.');
end

surface = isosurface(double(volume), threshold);
if isempty(surface.vertices) || isempty(surface.faces)
    error('No surface crosses the requested threshold; cannot export an empty mesh.');
end
vertices = 2 * (surface.vertices - 1) / (resolution - 1) - 1;

fileId = fopen(outputPath, 'w');
if fileId < 0
    error('Could not open mesh output path: %s', outputPath);
end
closeFile = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Octave voxel isosurface; vertices normalized to [-1,1]\n');
fprintf(fileId, 'v %.7f %.7f %.7f\n', vertices');
fprintf(fileId, 'f %d %d %d\n', surface.faces');
faces = surface.faces;
clear closeFile;
end