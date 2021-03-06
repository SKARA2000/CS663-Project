clear all;
path = "../Yale_Database/"; % Full face
m = 195;
n = 231;
num_persons = 15;
img_per_person = 11;
path_crop = "../yaleExpCropped";% Cropped face
m_crop = 50;
n_crop = 52;
%% Eigenfaces Accuracy using full faces on Yale Database
tic;
[acc_top30,acc_30] = Eigenfaces_method(path,m,n,num_persons,img_per_person,30,0);
fprintf("Accuracy on full faces using top 30 eigen vectors = %0.3f %%",acc_top30*100);
fprintf('\n');
fprintf("Accuracy on full faces using top 30 eigen vectors(excluding first 3) = %0.3f %%",acc_30*100);
toc;
%% Eigenfaces accuracy using cropped faces on Yale database
tic;
[acc_crop_top30,acc_crop_30] = Eigenfaces_method(path_crop,m_crop,n_crop,num_persons,img_per_person,30,0);
fprintf('\n');
fprintf("Accuracy on croppped faces using top 30 eigen vectors = %0.3f %%",acc_crop_top30*100);
fprintf('\n');
fprintf("Accuracy on croppped faces using top 30 eigen vectors(excluding first 3) = %0.3f %%",acc_crop_30*100);
fprintf('\n');
toc;
%% Eigenfaces accuracy after extrapolating onto different illumination directions
tic;
[accuracy,accuracy1] = Eigenfaces_Expt2_2(path,m,n,num_persons,img_per_person,30,0);
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
[acc_top30,~] = Eigenfaces_method(path,m,n,num_persons,img_per_person,163,1);
fprintf('\n');
fprintf("Accuracy on full faces using correlation = %0.3f %%",acc_top30*100);
fprintf('\n');
[acc_crop_top30,~] = Eigenfaces_method(path_crop,m_crop,n_crop,num_persons,img_per_person,163,1);
fprintf('\n');
fprintf("Accuracy on croppped faces using correlation= %0.3f %%",acc_crop_top30*100);
fprintf('\n');
[accuracy,~] = Eigenfaces_Expt2_2(path,m,n,num_persons,img_per_person,30,0);
fprintf('\n');
fprintf("Accuracy on full faces using correlation for centerlight= %0.3f %%",accuracy(1)*100);
fprintf('\n');
fprintf("Accuracy on full faces using correlation for rightlight= %0.3f %%",accuracy(4)*100);
fprintf('\n');
fprintf("Accuracy on full faces using correlation for leftlight= %0.3f %%",accuracy(7)*100);
fprintf('\n');
toc;
%% Linear Subspaces method Accuracy using full faces on Yale Database
tic;
[accuracy] = Linear_Subspace_Method(path,m,n,num_persons,img_per_person);
disp("Accuracy with full Faces is " +accuracy +"%");
toc;
%% Linear Subspaces method accuracy using cropped faces on Yale database
tic;
[accuracy] = Linear_Subspace_Method(path_crop,m_crop,n_crop,num_persons,img_per_person);
disp("Accuracy with cropped Faces is " +accuracy +"%");
toc;
%% Linear Subspaces method accuracy after extrapolating onto different illumination directions
tic;
[accuracy] = Linear_Subspace_Exp2_2(path,m,n,num_persons,img_per_person);
disp("Accuracy when centerlight is removed from Training set =" + accuracy(1));
disp("Accuracy when leftlight is removed from Training set =" + accuracy(4));
disp("Accuracy when rightlight is removed from Training set =" + accuracy(7));
toc;

