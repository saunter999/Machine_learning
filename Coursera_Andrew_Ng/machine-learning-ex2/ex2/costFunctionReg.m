function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

linarg=X*theta;
J=-y.*log(sigmoid(linarg))-(1.-y).*log(1.-sigmoid(linarg));
J=sum(J)/m;
J=J+(sum(theta.^2)-theta(1)^2)*lambda/(2.*m);
for j=1:n
 if j==1
    grad(j)=1./m* sum( (sigmoid(linarg)-y).*X(:,j) );
 else
    grad(j)=1./m* sum( (sigmoid(linarg)-y).*X(:,j) )+lambda/m*theta(j);
end






% =============================================================

end
