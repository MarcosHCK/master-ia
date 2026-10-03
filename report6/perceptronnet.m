% PERCEPTRONNET.M
% Train a 3-7-1 feedforward neural network using manual calculations.
%
% Architecture:
%   - 3 binary input features per sample
%   - 7 sigmoid neurons in one hidden layer; each receives all 3 inputs
%   - 1 sigmoid output neuron; it receives all 7 hidden outputs
%
% The eight input/target pairs below are the truth table supplied in the
% original perceptronnet.m. Run this script in MATLAB Online or Octave.

clear;
clc;
format short g;

%% Training data (preserved from the original file)
inputs = [0 0 0;
          0 0 1;
          0 1 0;
          0 1 1;
          1 0 0;
          1 0 1;
          1 1 0;
          1 1 1];
targets = [0; 1; 1; 1; 1; 0; 1; 1];

if size(inputs, 2) ~= 3 || size(inputs, 1) ~= numel(targets) || ...
        any((inputs(:) ~= 0) & (inputs(:) ~= 1)) || ...
        any((targets(:) ~= 0) & (targets(:) ~= 1))
    error('Training data must have three binary inputs and one binary target per row.');
end

%% Network setup
numInputs = 3;
numHidden = 7;
numOutputs = 1;
numSamples = size(inputs, 1);

% Fixed seed makes initialization and results reproducible.
rng(1, 'twister');
inputWeights = 2 * rand(numInputs, numHidden) - 1;
hiddenBiases = zeros(1, numHidden);
outputWeights = 2 * rand(numHidden, numOutputs) - 1;
outputBias = 0;

learningRate = 1.0;
maxEpochs = 30000;
epoch = 0;

%% Train with batch backpropagation and squared-error loss
% For each hidden neuron h:
%   hiddenSum(h) = x(1)*inputWeights(1,h) + ... + x(3)*inputWeights(3,h)
%                  + hiddenBiases(h)
%   hiddenOutput(h) = 1 / (1 + exp(-hiddenSum(h)))
% For the output neuron:
%   outputSum = hiddenOutput(1)*outputWeights(1) + ...
%               + hiddenOutput(7)*outputWeights(7) + outputBias
%   prediction = 1 / (1 + exp(-outputSum))
%
% The delta form below performs gradient descent on half squared error.
% All gradients for an epoch are accumulated before updating the weights.
for epoch = 1:maxEpochs
    inputWeightGradients = zeros(numInputs, numHidden);
    hiddenBiasGradients = zeros(1, numHidden);
    outputWeightGradients = zeros(numHidden, numOutputs);
    outputBiasGradient = 0;

    for sample = 1:numSamples
        hiddenSums = zeros(1, numHidden);
        hiddenOutputs = zeros(1, numHidden);

        % Input -> hidden: explicitly sum each input*weight product.
        for hidden = 1:numHidden
            for feature = 1:numInputs
                hiddenSums(hidden) = hiddenSums(hidden) + ...
                    inputs(sample, feature) * inputWeights(feature, hidden);
            end
            hiddenSums(hidden) = hiddenSums(hidden) + hiddenBiases(hidden);
            hiddenOutputs(hidden) = 1 / (1 + exp(-hiddenSums(hidden)));
        end

        % Hidden -> output: explicitly sum each hidden-output*weight product.
        outputSum = 0;
        for hidden = 1:numHidden
            outputSum = outputSum + hiddenOutputs(hidden) * outputWeights(hidden);
        end
        outputSum = outputSum + outputBias;
        prediction = 1 / (1 + exp(-outputSum));

        % Output delta and its contribution to output-layer gradients.
        outputDelta = (targets(sample) - prediction) * prediction * (1 - prediction);
        outputBiasGradient = outputBiasGradient + outputDelta;
        for hidden = 1:numHidden
            outputWeightGradients(hidden) = outputWeightGradients(hidden) + ...
                outputDelta * hiddenOutputs(hidden);
        end

        % Hidden deltas use the output weights before the epoch's update.
        for hidden = 1:numHidden
            hiddenDelta = hiddenOutputs(hidden) * (1 - hiddenOutputs(hidden)) * ...
                outputDelta * outputWeights(hidden);
            hiddenBiasGradients(hidden) = hiddenBiasGradients(hidden) + hiddenDelta;
            for feature = 1:numInputs
                inputWeightGradients(feature, hidden) = ...
                    inputWeightGradients(feature, hidden) + ...
                    hiddenDelta * inputs(sample, feature);
            end
        end
    end

    % Apply the mean gradient for the full training set.
    inputWeights = inputWeights + ...
        (learningRate / numSamples) * inputWeightGradients;
    hiddenBiases = hiddenBiases + ...
        (learningRate / numSamples) * hiddenBiasGradients;
    outputWeights = outputWeights + ...
        (learningRate / numSamples) * outputWeightGradients;
    outputBias = outputBias + ...
        (learningRate / numSamples) * outputBiasGradient;

    % Stop when all examples are correct and at least 0.05 from the threshold.
    correct = 0;
    confident = 0;
    for sample = 1:numSamples
        hiddenOutputs = zeros(1, numHidden);
        for hidden = 1:numHidden
            hiddenSum = 0;
            for feature = 1:numInputs
                hiddenSum = hiddenSum + ...
                    inputs(sample, feature) * inputWeights(feature, hidden);
            end
            hiddenSum = hiddenSum + hiddenBiases(hidden);
            hiddenOutputs(hidden) = 1 / (1 + exp(-hiddenSum));
        end

        outputSum = 0;
        for hidden = 1:numHidden
            outputSum = outputSum + hiddenOutputs(hidden) * outputWeights(hidden);
        end
        outputSum = outputSum + outputBias;
        prediction = 1 / (1 + exp(-outputSum));
        if double(prediction >= 0.5) == targets(sample)
            correct = correct + 1;
        end
        if (targets(sample) == 0 && prediction <= 0.45) || ...
                (targets(sample) == 1 && prediction >= 0.55)
            confident = confident + 1;
        end
    end

    if correct == numSamples && confident == numSamples
        break;
    end
end

%% Report learned parameters and evaluate the original data
fprintf('Manual 3-7-1 sigmoid network\n');
fprintf('Training examples: %d\n', numSamples);
fprintf('Hidden neurons: %d; output neurons: %d\n', numHidden, numOutputs);
fprintf('Training epochs: %d of %d\n', epoch, maxEpochs);
fprintf('Input-to-hidden weights (rows=input features, columns=hidden neurons):\n');
disp(inputWeights);
fprintf('Hidden biases:\n');
disp(hiddenBiases);
fprintf('Hidden-to-output weights:\n');
disp(outputWeights);
fprintf('Output bias: %.6f\n', outputBias);

correct = 0;
fprintf('\nPredictions for the supplied data:\n');
fprintf('Input       Target   Sigmoid output   Decision\n');
for sample = 1:numSamples
    hiddenOutputs = zeros(1, numHidden);
    for hidden = 1:numHidden
        hiddenSum = 0;
        for feature = 1:numInputs
            hiddenSum = hiddenSum + ...
                inputs(sample, feature) * inputWeights(feature, hidden);
        end
        hiddenSum = hiddenSum + hiddenBiases(hidden);
        hiddenOutputs(hidden) = 1 / (1 + exp(-hiddenSum));
    end

    outputSum = 0;
    for hidden = 1:numHidden
        outputSum = outputSum + hiddenOutputs(hidden) * outputWeights(hidden);
    end
    outputSum = outputSum + outputBias;
    prediction = 1 / (1 + exp(-outputSum));
    binaryPrediction = double(prediction >= 0.5);
    correct = correct + double(binaryPrediction == targets(sample));

    fprintf('[%d %d %d]       %d        %.6f          %d\n', ...
        inputs(sample, 1), inputs(sample, 2), inputs(sample, 3), ...
        targets(sample), prediction, binaryPrediction);
end
fprintf('Training accuracy: %d / %d (%.2f%%)\n', ...
    correct, numSamples, 100 * correct / numSamples);

%% Show a full manual forward-pass calculation for the first training row
sample = 1;
fprintf('\nManual forward-pass example for input [%d %d %d]:\n', ...
    inputs(sample, 1), inputs(sample, 2), inputs(sample, 3));
hiddenOutputs = zeros(1, numHidden);
for hidden = 1:numHidden
    hiddenSum = 0;
    fprintf('Hidden neuron %d: ', hidden);
    for feature = 1:numInputs
        product = inputs(sample, feature) * inputWeights(feature, hidden);
        hiddenSum = hiddenSum + product;
        fprintf('%d*%.6f', inputs(sample, feature), inputWeights(feature, hidden));
        if feature < numInputs
            fprintf(' + ');
        end
    end
    hiddenSum = hiddenSum + hiddenBiases(hidden);
    hiddenOutputs(hidden) = 1 / (1 + exp(-hiddenSum));
    fprintf(' + bias %.6f = %.6f; sigmoid = %.6f\n', ...
        hiddenBiases(hidden), hiddenSum, hiddenOutputs(hidden));
end

outputSum = 0;
fprintf('Output neuron: ');
for hidden = 1:numHidden
    outputSum = outputSum + hiddenOutputs(hidden) * outputWeights(hidden);
    fprintf('%.6f*%.6f', hiddenOutputs(hidden), outputWeights(hidden));
    if hidden < numHidden
        fprintf(' + ');
    end
end
outputSum = outputSum + outputBias;
finalOutput = 1 / (1 + exp(-outputSum));
fprintf(' + bias %.6f = %.6f; sigmoid = %.6f; binary decision = %d\n', ...
    outputBias, outputSum, finalOutput, double(finalOutput >= 0.5));