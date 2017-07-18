clear all;
images = loadMNISTImages('emnist-letters-train-images-idx3-ubyte');
labels = loadMNISTLabels('emnist-letters-train-labels-idx1-ubyte');

tr = [labels images'];

%prepare data for training network
n = size(tr, 1);                    
targets  = tr(:,1);                 
targetsd = dummyvar(targets);       
inputs = tr(:,2:end);               

inputs = inputs';                   
targets = targets';                 
targetsd = targetsd';              

rng(1);                             
c = cvpartition(n,'Holdout',n/3);   

Xtrain = inputs(:, training(c));    
Ytrain = targetsd(:, training(c));  
Xtest = inputs(:, test(c));         
Ytest = targets(test(c));           
Ytestd = targetsd(:, test(c));      

%Recurrent neural network

sweep = (50:50:250);                 
scores = zeros(length(sweep), 1);       
models = cell(length(sweep), 1);        
x = Xtrain;                             
t = Ytrain;                             

% scaled conjugate gradient
for i = 1:length(sweep)
    hiddenLayerSize = sweep(i);         
    net = layrecnet(1:2,hiddenLayerSize, 'trainscg'); 
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100; 
    net.divideParam.testRatio = 15/100; 
    net = train(net, x, t);             
    models{i} = net;                   
    p = net(Xtest);                    
    [~, p] = max(p);                    
    scores(i) = sum(Ytest == p) /length(Ytest)    
end

% plot accuracy figure 
figure
plot(sweep, scores, '.-')
xlabel('number of hidden neurons')
ylabel('categorization accuracy')
title('Number of hidden neurons vs. accuracy')