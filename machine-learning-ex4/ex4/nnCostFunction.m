function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% forward propogation
a1 = addBiasUnitToLayer(X);
[z2, a2] = getNextLayer(Theta1, a1);
a2 = addBiasUnitToLayer(a2);
[z3, a3] = getNextLayer(Theta2, a2);

%transform y to a mXk matrix
yy = zeros(m, num_labels);
for i = [1:m],
	yy(i, y(i)) = 1;
end;

% cost
% a2 is mXk matrix, k=num_labels, y is a mXk matrix

sum = 0;
for k = [1:num_labels],  %do for each label seperately
	yrow = yy(:, k)';
	hcol = a3(:, k);
	sum += -yrow*log(hcol) - (1-yrow) * log(1-hcol);
end;
J = sum/m;

%add regularization cost
ThetaNot1 = Theta1(:, 1);
ThetaNot2 = Theta2(:, 1);		
thetas = nn_params(:);

J += (lambda * ((thetas' * thetas) - ThetaNot1' * ThetaNot1 - ThetaNot2' * ThetaNot2))/(2*m);


%gradient
sum_error1 = zeros(size(Theta1));
sum_error2 = zeros(size(Theta2));
%error
for i = [1:m],
	error3 = a3(i, :) - yy(i, :); %1Xk
	error3 = error3'; %(kX1)	
	error2 = Theta2' * error3 .* a2(i, :)' .* (1-a2(i, :)'); %(prevLayerXk) * (kX1) = (prevLayerX1)
	
	error2 = error2(2:end);
	%fprintf('size1 [%d, %d] size2 [%d, %d] size3 [%d, %d]', size(sum_error2), size(error3), size(a2(i,:)));
	sum_error2 += error3 * a2(i, :); %(kX1)*(1Xhiddenlayersize)	 
	sum_error1 += error2 * a1(i, :); %(hiddenlayersizeX1) * (1XnumFeatures)

Theta1_grad = sum_error1/m;
Theta2_grad = sum_error2/m;

%regularization for gradient
Theta1_grad(:, 2:end) += (lambda * Theta1(:, 2:end))/m;
Theta2_grad(:, 2:end) += (lambda * Theta2(:, 2:end))/m;
% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
