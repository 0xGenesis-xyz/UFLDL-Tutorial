function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
z = cell(numHidden, 1);  %hidden
hAct{1} = data;  %input
for i=1:numHidden
    z{i} = bsxfun(@plus, stack{i}.W*hAct{i}, stack{i}.b);
    hAct{i+1} = sigmf(z{i}, [1 0]);
end

out = bsxfun(@plus, stack{numHidden+1}.W*hAct{numHidden+1}, stack{numHidden+1}.b);

numerator = exp(out);
denominator = sum(numerator, 1);
pred_prob = bsxfun(@rdivide, numerator, denominator);

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
m = size(data, 2);
one = zeros(m, ei.output_dim);
I = sub2ind(size(one), 1:size(one,1), labels');
one(I) = 1;

cost = (-1/m)*sum(sum(one'.*log(pred_prob)));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta = cell(numHidden+1, 1);  %hidden
delta{numHidden+1} = pred_prob-one';
for i=numHidden:-1:1
    delta{i} = stack{i+1}.W'*delta{i+1}.*(hAct{i+1}.*(1-hAct{i+1}));
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
for i=1:numHidden+1
    cost = cost+(ei.lambda/2)*sum(sum(stack{i}.W .^2));
end

for i=1:numHidden+1
    gradStack{i}.W = delta{i}*hAct{i}'/m+ei.lambda*stack{i}.W;
    gradStack{i}.b = sum(delta{i}, 2)/m;
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



