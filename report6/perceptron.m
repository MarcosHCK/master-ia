% PERCEPTRON.M
% Train a single sigmoid neuron to model a seven-vote weighted decision.
%
% To run it in MATLAB Online, paste it into a script (for example, perceptron.m)
% and press Run.
% Vote encoding: 1 = Yes, 0 = No. The output is converted to Yes/No at 0.5.
clear;
clc;
format short g;

%% Weighted-vote rule used to create the training examples
numVotes = 7;
voteWeights = [1 2 1 2 1 2 1];
yesThreshold = 6;

if numel(voteWeights) ~= numVotes || any(voteWeights <= 0) || ...
        any(isnan(voteWeights)) || any(isinf(voteWeights))
    error('voteWeights must contain seven finite, positive values.');
end
if yesThreshold <= 0 || yesThreshold > sum(voteWeights)
    error('yesThreshold must be greater than 0 and no greater than total vote weight.');
end

% Use every possible combination of seven binary votes (2^7 = 128 rows).
inputs = dec2bin(0:(2^numVotes - 1), numVotes) - '0';
targets = double(inputs * voteWeights(:) >= yesThreshold);

%% Train one sigmoid neuron using explicit sums and gradient updates
% Neuron: z = sum(input(i) * weight(i)) + bias
%         output = sigmoid(z) = 1 / (1 + exp(-z))
% Online squared-error update:
%         delta = (target - output) * output * (1 - output)
%         weight(i) = weight(i) + learningRate * delta * input(i)
%         bias = bias + learningRate * delta
weights = zeros(1, numVotes);
bias = 0;
learningRate = 0.5;
maxEpochs = 5000;
epoch = 0;

for epoch = 1:maxEpochs
    for sample = 1:size(inputs, 1)
        weightedSum = 0;
        for vote = 1:numVotes
            weightedSum = weightedSum + inputs(sample, vote) * weights(vote);
        end

        output = 1 / (1 + exp(-(weightedSum + bias)));
        delta = (targets(sample) - output) * output * (1 - output);

        for vote = 1:numVotes
            weights(vote) = weights(vote) + ...
                learningRate * delta * inputs(sample, vote);
        end
        bias = bias + learningRate * delta;
    end

    % Stop early as soon as every generated example is classified correctly.
    correct = 0;
    for sample = 1:size(inputs, 1)
        weightedSum = 0;
        for vote = 1:numVotes
            weightedSum = weightedSum + inputs(sample, vote) * weights(vote);
        end
        output = 1 / (1 + exp(-(weightedSum + bias)));
        prediction = double(output >= 0.5);
        if prediction == targets(sample)
            correct = correct + 1;
        end
    end
    if correct == size(inputs, 1)
        break;
    end
end

%% Report learned parameters and accuracy
fprintf('Sigmoid perceptron for a 7-vote weighted decision\n');
fprintf('Training examples: %d (all possible vote combinations)\n', size(inputs, 1));
fprintf('Reference vote weights: ');
fprintf('%.3g ', voteWeights);
fprintf('\nReference YES threshold: %.3g\n', yesThreshold);
fprintf('Training epochs: %d of %d\n', epoch, maxEpochs);
fprintf('Learned weights: ');
fprintf('%.6f ', weights);
fprintf('\nLearned bias: %.6f\n', bias);

correct = 0;
for sample = 1:size(inputs, 1)
    weightedSum = 0;
    for vote = 1:numVotes
        weightedSum = weightedSum + inputs(sample, vote) * weights(vote);
    end
    output = 1 / (1 + exp(-(weightedSum + bias)));
    prediction = double(output >= 0.5);
    if prediction == targets(sample)
        correct = correct + 1;
    end
end
fprintf('Training accuracy: %d / %d (%.2f%%)\n', correct, size(inputs, 1), ...
    100 * correct / size(inputs, 1));

%% Show the actual input*weight terms, sum, sigmoid, and binary decision
% Show examples from both classes, including the first and last of each.
exampleRows = unique([find(targets == 0, 1, 'first'), ...
                      find(targets == 0, 1, 'last'), ...
                      find(targets == 1, 1, 'first'), ...
                      find(targets == 1, 1, 'last')]);
fprintf('\nExample calculations using the learned neuron:\n');
for example = 1:numel(exampleRows)
    sample = exampleRows(example);
    weightedSum = 0;

    fprintf('\nVotes [');
    fprintf('%d ', inputs(sample, :));
    fprintf(']\nProducts: ');
    for vote = 1:numVotes
        product = inputs(sample, vote) * weights(vote);
        weightedSum = weightedSum + product;
        fprintf('%d*%.6f = %.6f', inputs(sample, vote), weights(vote), product);
        if vote < numVotes
            fprintf(' + ');
        end
    end

    z = weightedSum + bias;
    output = 1 / (1 + exp(-z));
    prediction = double(output >= 0.5);
    if prediction == 1
        predictionLabel = 'YES';
    else
        predictionLabel = 'NO';
    end
    if targets(sample) == 1
        targetLabel = 'YES';
    else
        targetLabel = 'NO';
    end
    fprintf('\nWeighted sum = %.6f; bias = %.6f; z = %.6f\n', ...
        weightedSum, bias, z);
    fprintf('Sigmoid(z) = 1 / (1 + exp(-z)) = %.6f\n', output);
    fprintf('Prediction: %s; reference target: %s\n', ...
        predictionLabel, targetLabel);
end

%% Apply the trained neuron to your own seven-vote input here
% Replace this example with any seven values containing only 0 or 1.
newVotes = [1 0 1 1 0 1 0];
if numel(newVotes) ~= numVotes || any((newVotes ~= 0) & (newVotes ~= 1))
    error('newVotes must contain exactly seven binary values (0 or 1).');
end

newWeightedSum = 0;
for vote = 1:numVotes
    newWeightedSum = newWeightedSum + newVotes(vote) * weights(vote);
end
newOutput = 1 / (1 + exp(-(newWeightedSum + bias)));
if newOutput >= 0.5
    newDecision = 'YES';
else
    newDecision = 'NO';
end
fprintf('\nYour input [');
fprintf('%d ', newVotes);
fprintf('] -> weighted sum %.6f, sigmoid output %.6f, decision %s\n', ...
    newWeightedSum, newOutput, newDecision);
