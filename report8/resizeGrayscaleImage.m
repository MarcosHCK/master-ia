function resizedImage = resizeGrayscaleImage(image, outputRows, outputColumns)
%RESIZEGRAYSCALEIMAGE Resize a grayscale image with core-Octave interp2.

if ~isnumeric(image) || ~isreal(image) || ndims(image) ~= 2 || ...
        isempty(image) || any(~isfinite(image(:)))
    error('image must be a non-empty, finite, real grayscale matrix.');
end
if ~isscalar(outputRows) || ~isscalar(outputColumns) || ...
        outputRows < 1 || outputColumns < 1 || ...
        outputRows ~= floor(outputRows) || outputColumns ~= floor(outputColumns)
    error('Output dimensions must be positive integers.');
end

image = double(image);
if isequal(size(image), [outputRows, outputColumns])
    resizedImage = image;
    return;
end
if size(image, 1) < 2 || size(image, 2) < 2
    error('The source image must have at least two rows and two columns.');
end

queryX = linspace(1, size(image, 2), outputColumns);
queryY = linspace(1, size(image, 1), outputRows);
[queryGridX, queryGridY] = meshgrid(queryX, queryY);
resizedImage = interp2(image, queryGridX, queryGridY, 'linear', 0);
resizedImage = min(1, max(0, resizedImage));
end