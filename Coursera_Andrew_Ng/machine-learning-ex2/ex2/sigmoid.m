function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
col=columns(z);row=rows(z);

%scalar case
if col==1&&row==1
   g=1./(1.+exp(-z));
end
%matrix case
if col>1&&row>1
   for i=1:row
       g(i,:)=1./(1.+exp(-z(i,:)));
   end
end
%vector case
if col==1 || row==1
   if row==1
      g(1,:)=1./(1.+exp(-z(1,:)));
   else
      g(:,1)=1./(1.+exp(-z(:,1)));
   end
end




% =============================================================

end
