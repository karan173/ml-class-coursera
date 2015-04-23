function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

linear_hypothesis = X*theta; 
logistic_hypothesis = sigmoid(linear_hypothesis); % (mX1)
cost1 = y' * log(logistic_hypothesis); % (1Xm)*(mX1) = (1X1) 
cost2 = (1-y') * log(1-logistic_hypothesis);
cost3 = (lambda/2.0) * ((theta' * theta) - (theta(1)^2)); %we dont want to regularize thetha0
J = (-cost1 -cost2 + cost3)/m;

hypothesis_error = logistic_hypothesis-y;
grad = (X' * hypothesis_error)/m; % (nXm)*(mX1) = (nX1)
%add regularization parameters to all except first
grad(2:end) += (lambda/m)*theta(2:end);
% =============================================================

end
