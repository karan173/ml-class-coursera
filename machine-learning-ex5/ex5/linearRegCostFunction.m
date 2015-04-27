function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X is mXn, theta is nX1
hypothesis = X * theta; %this is mX1
hyp_error = hypothesis-y;
theta_rem = theta(2:end);
theta_rem = theta_rem(:);
J = ((hyp_error' * hyp_error) + (lambda * (theta_rem' * theta_rem)))/(2.0*m);

% =========================================================================
grad = (X' * hyp_error)/m; %  (nXm) * (mX1) = (nX1)
grad(2:end) += (lambda * theta_rem)/m;
end
