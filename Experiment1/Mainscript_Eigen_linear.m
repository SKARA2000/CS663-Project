clear all;
m = 168;
n = 192;
num_person = 38;
path = "../CroppedYale";
K=250;
[set1,set2,set3,set4,set5,setcounter] = Create_Subsets(m,n,path);
%% Eigenfaces Result for training on Set 1 and 5 and extrapolating
tic;
[accuracy, accuracy1] = Eigenfaces_method(set1,set2,set3,set4,set5,setcounter,m,n,250,num_person,0);
disp("Accuracy as illumination angle increases for top 250");
disp(accuracy);
disp("Accuracy as illumination angle increases for top 250(excluding top 3)");
disp(accuracy1);
toc;
%% Correlation result for the same
tic;
[accuracy, ~] = Eigenfaces_method(set1,set2,set3,set4,set5,setcounter,m,n,566,num_person,1);
disp("Accuracy as illumination angle increases for correlation");
disp(accuracy);
toc;
%% Eigenfaces Result for training on only Set 1 and extrapolating
tic;
[accuracy, accuracy1] = Eigenfaces_method2(set1,set2,set3,set4,set5,setcounter,m,n,140,num_person,0);
disp("Accuracy as illumination angle increases for top 140");
disp(accuracy);
disp("Accuracy as illumination angle increases for top 140(excluding top 3)");
disp(accuracy1);
toc;
%% Correlation result for the same
tic;
[accuracy, ~] = Eigenfaces_method2(set1,set2,set3,set4,set5,setcounter,m,n,566,num_person,1);
disp("Accuracy as illumination angle increases for correlation");
disp(accuracy);
toc;
%% Linear Subspaces Result for training on Set 1 and 5 and extrapolating
tic;
accuracy = Linear_Subspace_Method(set1,set2,set3,set4,set5,setcounter,m,n,num_person);
disp("Accuracy as illumination angle increases ");
disp(accuracy);
toc;
%% Linear Subspaces Result for training only on Set 1 and extrapolating
tic;
accuracy = Linear_Subspace_Method1(set1,set2,set3,set4,set5,setcounter,m,n,num_person);
disp("Accuracy on the Subset 2 is " + accuracy(1) + "%");
disp("Accuracy on the Subset 3 is " + accuracy(2) + "%");
disp("Accuracy on the Subset 4 is " + accuracy(3) + "%");
toc;