%% Clear
clear all;
tic;
%% Function LinearSubspace Method
%path = "../yaleExpCropped";% Cropped face
%m = 50;
%n=52;
path = "../Yale_Database"; % Full face
m = 195;
n = 231;
num_persons = 15;
img_per_person = 11;
N = 15*11;
images = dir(path);
accuracy = 0;
for i = 1:1:num_persons   %(i,j) : Test Image
    for j = 1:1:img_per_person
        Eigen_Vec_3 = zeros([m*n,3,num_persons]);
        %Mean_Vec = zeros([m*n,num_persons]);
        for k = 1:1:num_persons  %(k,l) : Training Image using Leaving one out strategy
            if i == k
                X_train = zeros([m*n,img_per_person-1]);
                count = 0;
                for l = 1:1:img_per_person
                    if l == j
                    else
                        count = count+1;
                        X_train(:,count) = im2double(reshape(imread(path +"/" +images(((k-1)*img_per_person) + l + 2).name),[m*n,1]));
                    end
                end
                %mean = sum(X_train,2)/(img_per_person-1);
                %X_train = X_train - mean;
                %Mean_Vec(:,k) = mean;
                L = X_train'*X_train/(img_per_person-2);
                [U, S, ~] = svd(L);
                V = X_train*U;
                normv = sqrt(sum(V.^2,1)); 
                V = bsxfun(@rdivide,V,normv);
                Eigen_Vec_3(:,:,k) = V(:,1:3);
            else
                X_train = zeros([m*n,img_per_person]);
                for l = 1:1:img_per_person
                    X_train(:,l) = im2double(reshape(imread(path +"/" +images(((k-1)*img_per_person) + l + 2).name),[m*n,1]));
                end
                %mean = sum(X_train,2)/img_per_person;
                %X_train = X_train - mean;
                %Mean_Vec(:,k) = mean;
                L = X_train'*X_train/(img_per_person-1);
                [U, S, ~] = svd(L);
                V = X_train*U;
                normv = sqrt(sum(V.^2,1)); 
                V = bsxfun(@rdivide,V,normv);
                Eigen_Vec_3(:,:,k) = V(:,1:3); 
            end
        end
        
        %Testing
        TestImg = im2double(reshape(imread(path +"/" +images(((i-1)*img_per_person) + j + 2).name),[m*n,1]));
        distance = zeros([num_persons,1]);
        for k = 1:1:num_persons
            X_test = TestImg;
            Test_Coeff = Eigen_Vec_3(:,:,k)'*X_test;
            Reconstruction = Eigen_Vec_3(:,:,k)*Test_Coeff;
            distance(k) = sum((X_test - Reconstruction).^2);                
        end
        [~, indx] = min(distance);
        if indx == i
            accuracy = accuracy + 1;
        end
    end 
end   
accuracy = (accuracy/N)*100;
disp("Accuracy with full Faces is " +accuracy +"%");
%disp("Accuracy with Cropped Faces is " +accuracy +"%");
toc;