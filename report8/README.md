# Grayscale image to surface reconstruction (MATLAB / Octave)

This toolbox-free, hand-coded 2-D convolutional neural network maps one square
grayscale image to one square height map. Its prediction is a **surface**, not
a volumetric object. By default its encoding is `height = intensity - 0.5`:
50% gray is height zero, 20% is -0.3, and 70% is +0.2. Inference supports an
optional nonnegative height scale: `height = heightScale * (intensity - 0.5)`.
Heights are relative and measured in normalized grayscale units multiplied by
that scale.

The scripts use MATLAB-compatible syntax and base Octave 11.3 functions; no
additional Octave packages, Deep Learning Toolbox, or image-processing toolbox
are required. Octave's built-in `imread`/`imwrite` are available in the tested
environment. For MATLAB, saving version `-v7` MAT files is supported.

## Quick start

From the Octave prompt:

```matlab
cd('/home/marcos/Documents/projects/master/ia/report8')
addpath(pwd)
generate_surface_dataset();              % 320 smooth 64-by-64 examples
network = train_surface_network();       % early stopping; saves model weights
[heightMap, files] = reconstruct_surface('my-image.png');
```

The dataset and trained weights default to `examples.mat` and
`trained-network.mat` in this directory. Supply a square grayscale image with
the model's dimensions. `reconstruct_surface` accepts either a filename (also
RGB, converted to grayscale) or a numeric image matrix. Integer image pixels
are normalized by their type's maximum; floating-point pixels must already be
in `[0, 1]`. Image geometry must exactly match `N` (default 64); images are not
silently resized.

## Dataset options

```matlab
% Default: N=64, 320 examples, Gaussian image noise sigma=0.025, seed=1.
dataset = generate_surface_dataset();

% Larger clean dataset at N=64.
dataset = generate_surface_dataset('clean-examples.mat', 500, 64, 0, 22);

% Smaller noisy dataset at N=32, useful for quick development tests.
dataset = generate_surface_dataset('small-examples.mat', 24, 32, 0.04, 7);
```

Each clean target is formed from a smooth low-frequency wave plus several
smooth Gaussian hills and depressions. Before noise is added, each grayscale
input pixel is wrapped with the report 7 modulo transform at threshold 0.5;
the wrapped values are shifted back into `[0,1]` grayscale. Noise is added to
these wrapped inputs, not to the clean targets. Use `noiseStd = 0` for noiseless
examples. The output MAT file contains a struct `dataset` with `N`, `noiseStd`,
`wrapThreshold`, `seed`, and `train`,
`validation`, and `test` splits. Each split contains `inputImages`,
`targetHeights` (relative heights in `[-0.5,0.5]`), and `cleanImages`. Splitting
is done between whole surfaces to avoid leakage between pixels of one example.

## Training options

```matlab
network = train_surface_network();
network = train_surface_network('small-examples.mat', 'small-model.mat', ...
    0.003, 100, 10, 1e-5);
```

Arguments are dataset file, output model file, Adam learning rate, maximum
epochs, early-stopping patience, minimum validation-RMSE improvement, and
mini-batch size. Defaults are 0.003, 200, 20, `1e-5`, and 16, respectively.
Training uses the validation split for stopping and returns the weights from
the best observed validation checkpoint. The saved file contains variable
`network`, including the model parameters and training/validation RMSE
histories. The test split is not used for fitting or checkpoint selection; its
RMSE (`network.testRMSE`) is computed once with the selected best model for a
held-out performance estimate.

## Inference and mesh output

```matlab
[heightMap, files] = reconstruct_surface('my-image.png');
% Or pass a normalized numeric N-by-N image and explicit model/output paths:
[heightMap, files] = reconstruct_surface(imageMatrix, ...
    'trained-network.mat', 'sample');
% Exaggerate elevations and depressions by a factor of 3:
[heightMap, files] = reconstruct_surface(imageMatrix, ...
    'trained-network.mat', 'sample-x3', 3);
```

The optional fourth argument `heightScale` defaults to `1` and must be a
finite nonnegative scalar. It scales the returned height map and the OBJ
vertex heights; predicted grayscale intensities remain unchanged. The chosen
scale is also saved in the MAT output.

Outputs are `sample-prediction.mat` (variables `heightMap`,
`predictedIntensity`, and `inputImage`) and `sample-prediction.obj`. The OBJ
contains `N*N` grid vertices and `2*(N-1)*(N-1)` triangular faces: a single
open, connected surface with `x`/`y` grid coordinates and predicted height as
`z`. It has no bottom, side walls, or enclosed volume. The returned height map
has zero at level gray and is the appropriate output for further processing.

All default MAT and OBJ artifacts are ignored by Git in this directory.