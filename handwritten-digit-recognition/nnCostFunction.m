function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta1_x = Theta1(:,(2:end));   %remove theta1(0)
Theta2_x = Theta2(:,(2:end));   %remove theta2(0)
regterm = [Theta1_x(:);Theta2_x(:)]'*[Theta1_x(:);Theta2_x(:)];

class_y = zeros(m,num_labels);
for i = 1:num_labels
    class_y(:,i) = y==i;
end

%Forward propagation
a1 = [ones(m,1),X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2*Theta2';
h = sigmoid(z3);

%Compute cost
J = -((class_y(:)'*(log(h(:)))) + ((1-class_y(:))'*(log(1-h(:))))-(lambda*regterm/2))/m;

%Back propagation for gradient
for i = 1:m
    delta3(i,:) = h(i,:)-class_y(i,:);
    Theta2_grad = Theta2_grad+delta3(i,:)'*a2(i,:);
    delta2(i,:) = (delta3(i,:)*Theta2_x).*sigmoidGradient(z2(i,:));
    Theta1_grad = Theta1_grad+delta2(i,:)'*a1(i,:);
end

Theta1(:,1) = 0;
Theta2(:,1) = 0;

% Unroll gradients
grad = ([Theta1_grad(:);Theta2_grad(:)]+lambda*[Theta1(:);Theta2(:)])/m;

end
