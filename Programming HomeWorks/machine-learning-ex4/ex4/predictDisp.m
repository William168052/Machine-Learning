function predictDisp(X,Theta1,Theta2)

for idx = 1:100
randNum = unidrnd(size(X,1));
displayData(X(randNum,:));
pred = predict(Theta1, Theta2, X(randNum,:));
fprintf('result:%d\n',pred);
pause;
end
