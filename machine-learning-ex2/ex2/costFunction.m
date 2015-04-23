function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

linear_hypothesis = X*theta; 
logistic_hypothesis = sigmoid(linear_hypothesis); % (mX1)
cost1 = y' * log(logistic_hypothesis); % (1Xm)*(mX1) = (1X1) 
cost2 = (1-y') * log(1-logistic_hypothesis);
J = (-cost1 -cost2)/m;

hypothesis_error = logistic_hypothesis-y;
grad = (X' * hypothesis_error)/m; % (nXm)*(mX1) = (nX1)
% =============================================================

end
