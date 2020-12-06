clear all;
m = 168;
n = 192;
num_persons = 38;
path = "../CroppedYale";


%% Eigenfaces accuracy using cropped faces on Yale database
tic;
[acc_crop_top30,acc_crop_30] = Eigenfaces_method(path,m,n,num_persons,250,0);
fprintf('\n');
fprintf("Accuracy on croppped faces using top 250 eigen vectors = %0.3f %%",acc_crop_top30*100);
fprintf('\n');
fprintf("Accuracy on croppped faces using top 250 eigen vectors(excluding first 3) = %0.3f %%",acc_crop_30*100);
fprintf('\n');
toc;
%% Eigenfaces accuracy after extrapolating onto different illumination directions
tic;
[accuracy,accuracy1] = Eigenfaces_Expt2_2(path,m,n,num_persons,250,0);
fprintf('\n');
fprintf("Accuracy on full faces using top 30 eigen vectors for centerlight= %0.3f %%",accuracy(1)*100);
fprintf('\n');
fprintf("Accuracy on full faces using top 30 eigen vectors for rightlight= %0.3f %%",accuracy(4)*100);
fprintf('\n');
fprintf("Accuracy on full faces using top 30 eigen vectors for leftlight= %0.3f %%",accuracy(7)*100);
fprintf('\n');
fprintf("Accuracy on full faces using top 30 eigen vectors for centerlight(excluding top 3)= %0.3f %%",accuracy1(1)*100);
fprintf('\n');
fprintf("Accuracy on full faces using top 30 eigen vectors for rightlight(excluding top 3)= %0.3f %%",accuracy1(4)*100);
fprintf('\n');
fprintf("Accuracy on full faces using top 30 eigen vectors for leftlight(excluding top 3)= %0.3f %%",accuracy1(7)*100);
fprintf('\n');
toc;
%% Correlation results for the same
tic;
[acc_crop_top30,~] = Eigenfaces_method(path,m,n,num_persons,163,1);
fprintf('\n');
fprintf("Accuracy on croppped faces using correlation= %0.3f %%",acc_crop_top30*100);
fprintf('\n');
[accuracy,~] = Eigenfaces_Expt2_2(path,m,n,num_persons,30,0);
fprintf('\n');
fprintf("Accuracy on full faces using correlation for centerlight= %0.3f %%",accuracy(1)*100);
fprintf('\n');
fprintf("Accuracy on full faces using correlation for rightlight= %0.3f %%",accuracy(4)*100);
fprintf('\n');
fprintf("Accuracy on full faces using correlation for leftlight= %0.3f %%",accuracy(7)*100);
fprintf('\n');
toc;
