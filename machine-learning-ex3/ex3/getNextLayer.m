function a = getNextLayer(theta, X)
% 	Get next layer for a neural network
%	Assumes bias unit has not yet been added

%theta has dimensions size_new_layer X size_old_layer
%X has dimensions m X size_old_layer

	newX = [ones(size(X,1), 1) X]; %add bias unit
	a = sigmoid(newX * theta'); %(m X size_old_layer) * (size_old_layer X size_new_layer)
								%=(mXsize_new_layer)
end