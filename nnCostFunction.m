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


a1 = [ones(1,m); X']; % Add a row of 1's to the X' matrix

% output from input layer
for i=1:m 
  
    z2(:,i)=Theta1*a1(:,i); % size(Theta1)=25*401   size(Xt(:,1))= 401*1  >  size(z_2(:,i)=25*1) >  size(z_2(:,:)=25*5000)
  
    a2(:,i) = sigmoid(z2(:,i));  %size(a_2(:,:)=25*5000
end


                       
a2 = [ones(1,m); a2]; % Add a row of 1's to the a2 matrix > size(a_2(:,:)=26*5000

% output from hidden layer
for i=1:m 
                 
     z3(:,i)=Theta2*a2(:,i); %  size(Theta2)=10*26   size(a_2(:,i)=26*1 >  size(z_3(:,i)=10*1) >  size(z_3(:,:)=10*5000)
  
    h_theta(:,i) = sigmoid(z3(:,i)); 
end


% recod labels to identity matrix for the purpose of training the nueral network
ynn = zeros(num_labels, m); % size(ynn(:,:)=10*5000)


for i = 1:m
    ynn(y(i), i) = 1; 
end

%unregularized cost function

%J=1/m*sum(-ynn(:).*log(h_theta(:))-(1-ynn(:)).*log(1-h_theta(:)));


%regularized cost function

Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);


J = 1/m*sum(-ynn(:).*log(h_theta(:))-(1-ynn(:)).*log(1-h_theta(:)))...
  + lambda/2/m*(sum(Theta1NoBias(:).*Theta1NoBias(:))+sum(Theta2NoBias(:).*Theta2NoBias(:)));
  

% Compute gradient for Theta1 and Theta2 with back propagation
  
  %error from output layer
  
  % Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

for i=1:m 
  
    delta3(:,i)= h_theta(:,i)-ynn(:,i); %  size(delta_3(:,:)=10*5000)
end


%error from hidden layer 
  
delta2= Theta2'*delta3;  %  size(Theta2')=26*10   >  size(delta2(:,:)=26*5000)
    
delta2=delta2(2:end,:);  %size(delta2(:,:)=25*5000)
    
delta2=delta2.*sigmoidGradient(z2);

    
% gradient accumulation for input layer 

Delta1=delta2*a1';    %size(Delta_1(:,:)=25*400  same size as Theta1-1


% gradient accumulation for hidden layer 
   
Delta2=delta3*a2';   %size(Delta_2(:,:)=10*25  same size as Theta2-1

    
%unregularized gradients

grad = [Delta1(:); Delta2(:)]/m;


%unregularized gradients
Theta1(:,1) = 0;
Theta2(:,1) = 0;

grad = [Delta1(:); Delta2(:)]/m + lambda*[Theta1(:); Theta2(:)]/m;


end
