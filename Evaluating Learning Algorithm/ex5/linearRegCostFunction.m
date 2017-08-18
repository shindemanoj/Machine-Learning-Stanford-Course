function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

Hx = X * theta;

J = sum(sum((Hx - y).^ 2));
J = J / (2 * m);

reg = theta.^2;
reg(1) = 0;
reg = (lambda / (2*m)) * sum(sum(reg));

J = J + reg;

grad = X' * (Hx - y);
grad = grad ./ m;

tempTh = theta;
tempTh(1) = 0;
tempTh = (lambda / m)*tempTh;

grad = grad + tempTh;

% =========================================================================

grad = grad(:);

end
