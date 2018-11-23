function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a_1=[ones(m,1),X];   %The first layer a_1 dimension is (m,n+1) where n=400
z_2=Theta1*a_1';     %z_2 dimension is (25,m)
a_2=sigmoid(z_2);    %a_2 dimension is (25,m) 
a_2=a_2';            %a_2 dimension is (m,25)
a_2=[ones(m,1),a_2]; %The hidden layer a_2 dimension is (m,26)
z_3=Theta2*a_2';     %z_3 dimension is (10,m)
a_3=sigmoid(z_3);    %a_3 dimension is (10,m)
a_3=a_3';            %a_3 dimension is (m,10)

%for i=1:m
%   [val,p(i)]=max(a_3(i,:));
%end
[val,p]=max(a_3,[],2);

% =========================================================================


end
