function faces = exportSurfaceMesh(heightMap, outputPath)
%EXPORTSURFACEMESH Write a sampled height field as a triangular OBJ mesh.

if ~isnumeric(heightMap) || ~isreal(heightMap) || ndims(heightMap) ~= 2 || ...
        min(size(heightMap)) < 2 || any(~isfinite(heightMap(:))) || ...
        any(heightMap(:) < 0) || any(heightMap(:) > 1)
    error('heightMap must be a finite 2-D array in [0,1] with at least 2 rows and columns.');
end
if ~ischar(outputPath) || isempty(outputPath)
    error('outputPath must be a non-empty character vector.');
end

[rows, columns] = size(heightMap);
[worldX, worldY] = meshgrid(linspace(-1, 1, columns), linspace(-1, 1, rows));
vertices = [worldX(:), worldY(:), 2 * double(heightMap(:)) - 1];
faces = zeros(2 * (rows - 1) * (columns - 1), 3);
faceIndex = 1;
for column = 1:columns - 1
    for row = 1:rows - 1
        lowerLeft = (column - 1) * rows + row;
        lowerRight = lowerLeft + rows;
        upperLeft = lowerLeft + 1;
        upperRight = lowerRight + 1;
        faces(faceIndex, :) = [lowerLeft, lowerRight, upperRight];
        faces(faceIndex + 1, :) = [lowerLeft, upperRight, upperLeft];
        faceIndex = faceIndex + 2;
    end
end

fileId = fopen(outputPath, 'w');
if fileId < 0
    error('Could not open mesh output path: %s', outputPath);
end
closeFile = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Height-field surface; x/y and z normalized to [-1,1]\n');
fprintf(fileId, 'v %.7f %.7f %.7f\n', vertices');
fprintf(fileId, 'f %d %d %d\n', faces');
clear closeFile;
end