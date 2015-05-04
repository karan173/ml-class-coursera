function net_error = trainAndFindError(X, y, Xval, yval, C, sigma)
% Trains an SVM for the given parameters and finds the fractional error on the cross validation set

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
pred = svmPredict(model, Xval);
net_error = getClassificationError(pred, yval);
end;
