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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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



%将y改成向量输出形式
y_vect = zeros(num_labels,m);  % 10 * 5000
for i = 1:m
	y_vect(y(i),i) = 1;
end


%计算h(x)
a1 = X;
a1 = [ones(size(X,1),1),X];  % 5000 * 401
z2 = Theta1 * a1';  % 25 * 5000
a2 = sigmoid(z2); 
a2 = [ones(1,size(a2,2));a2]; % 26 * 5000
z3 = Theta2 * a2;
a3 = sigmoid(z3);  % 10 * 5000
h_x = a3;


for i = 1:m 
	J = J + sum(- y_vect(:,i) .* log(h_x(:,i))) - sum((1 - y_vect(:,i)) .* log(1 - h_x(:,i)));
end

J = J / m;

% 正则化
regularization = lambda / (2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + regularization;


% 反向传播

Delta1 = zeros(size(Theta1));  % 25 * 401
Delta2 = zeros(size(Theta2));  % 10 * 26

for i = 1:m
	delta_3 = a3(:,i) - y_vect(:,i);  % 10 * 1
	delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z2(:,i));  
	Delta2 = Delta2 + delta_3 * a2(:,i)';
	Delta1 = Delta1 + delta_2 * a1(i,:);
end

Theta1(:,1) = zeros(size(Theta1,1),1);
Theta2(:,1) = zeros(size(Theta2,1),1);

Theta1_grad = Delta1 / m + (lambda / m) * Theta1;
Theta2_grad = Delta2 / m + (lambda / m) * Theta2;	
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
