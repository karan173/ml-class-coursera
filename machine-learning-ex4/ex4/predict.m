function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a1 = addBiasUnitToLayer(X);
[z2, a2] = getNextLayer(Theta1, a1);
a2 = addBiasUnitToLayer(a2);
[z3, a3] = getNextLayer(Theta2, a2);

[dummy, p] = max(a3, [], 2);

% =========================================================================


end
