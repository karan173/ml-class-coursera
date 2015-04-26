function a = addBiasUnitToLayer(X)
a = [ones(size(X,1), 1) X];
end;