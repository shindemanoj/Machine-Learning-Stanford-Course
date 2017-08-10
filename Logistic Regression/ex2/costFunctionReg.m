function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
  J = J + (-y(i) * log(sigmoid(X(i, :)*theta)) - (1 - y(i)) * ...
  (log(1 - sigmoid(X(i, :)*theta))));
  
  for j = 1:size(X,2)
    if (j == 1)
      grad(j) = grad(j) + (sigmoid(X(i, :)*theta) - y(i)) * X(i, j);
    else
      grad(j) = grad(j) + (sigmoid(X(i, :)*theta) - y(i)) * X(i, j) + ...
      ((lambda/m)*theta(j));
    endif
  end
  
end
grad = grad ./ m;
J = J / m;

regparam = 0;

for j=2:size(theta)
  regparam = regparam + (theta(j) ^ 2);
end 

regparam = (lambda / (2 * m)) * regparam;
J = J + regparam;
  
% =============================================================

end
