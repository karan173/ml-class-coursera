function classification_error = getClassificationError(pred, yval)
% Returns the fractional classification error where pred is the vector of predicted classes and yval
% is the vector of labels

classification_error = mean(pred != yval);
end