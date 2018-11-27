function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
valls=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
nval=length(valls);
error_val=zeros(nval);
error_val=error_val(:);
it=1;
for i=1:nval
    C=valls(i);
    for j=1:nval
        sigma=valls(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	predictions=svmPredict(model,Xval);
	error_val(it)=mean(double(predictions~=yval));
	it=it+1;
    end
end

[err,opmind]=min(error_val);

sind=mod(opmind,nval);
if sind==0
sind=nval;
Cind=opmind/nval;
else
Cind=(opmind-sind)/nval+1;
end
C=valls(Cind);
sigma=valls(sind);
C
sigma

% =========================================================================

end
