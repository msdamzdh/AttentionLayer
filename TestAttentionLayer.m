%% Test attention layer
clc
close all
clear
%%
X1=dlarray(rand(50,20,10),"CBT");
X2=dlarray(rand(30,20,10),"CTB");
X3=dlarray(rand(40,20,10),"BCT");
X4=dlarray(rand(27,14,10),"BTC");
X5=dlarray(rand(15,10,12),"TCB");
X6=dlarray(rand(16,9,10),"TBC");
%%
sz1 = size(X1);
sz2 = size(X2);
sz3 = size(X3);
sz4 = size(X4);
sz5 = size(X5);
sz6 = size(X6);
InputDim1 = sz1(finddim(X1,"C"));
InputDim2 = sz2(finddim(X2,"C"));
InputDim3 = sz3(finddim(X3,"C"));
InputDim4 = sz4(finddim(X4,"C"));
InputDim5 = sz5(finddim(X5,"C"));
InputDim6 = sz6(finddim(X6,"C"));
InputFormat1 = dims(X1);
InputFormat2 = dims(X2);
InputFormat3 = dims(X3);
InputFormat4 = dims(X4);
InputFormat5 = dims(X5);
InputFormat6 = dims(X6);
%% Defining nuber of heads
NumberOfHead = 8;
QueryDim = 40; % Should be divisible by NumberOfHead
ValuDim = 56; % Should be divisible by NumberOfHead
OutputDim = 60;
%%
layer1 = attentionLayer(InputDim1,QueryDim,ValuDim,OutputDim,NumberOfHead,InputFormat1);
layer2 = attentionLayer(InputDim2,QueryDim,ValuDim,OutputDim,NumberOfHead,InputFormat2);
layer3 = attentionLayer(InputDim3,QueryDim,ValuDim,OutputDim,NumberOfHead,InputFormat3);
layer4 = attentionLayer(InputDim4,QueryDim,ValuDim,OutputDim,NumberOfHead,InputFormat4);
layer5 = attentionLayer(InputDim5,QueryDim,ValuDim,OutputDim,NumberOfHead,InputFormat5);
layer6 = attentionLayer(InputDim6,QueryDim,ValuDim,OutputDim,NumberOfHead,InputFormat6);
%%
Z1 = predict(layer1,X1);
Z2 = predict(layer2,X2);
Z3 = predict(layer3,X3);
Z4 = predict(layer4,X4);
Z5 = predict(layer5,X5);
Z6 = predict(layer6,X6);