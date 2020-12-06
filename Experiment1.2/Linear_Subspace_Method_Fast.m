%% Clear
clear all;
tic;
%% Function LinearSubspace Method
%function [accuracy] = LinearSubspace(path)
    path = "../Yale_Database"; 
    m = 195;
    n = 231;
    num_persons = 15;
    img_per_person = 11;
    N = 15*11;
    images = dir(path);
    accuracy = 0;
    Eigen_Vec_3 = zeros([m*n,3,num_persons]);
    Mean_Vec = zeros([m*n,num_persons]);

    for i = 1:1:num_persons
        X_train = zeros([m*n,img_per_person]);
        for j = 1:1:img_per_person
            X_train(:,j) = im2double(reshape(imread(path +"/" +images(((i-1)*img_per_person) + j + 2).name),[m*n,1]));
        end
        %mean = sum(X_train,2)/img_per_person;
        %X_train = X_train - mean;
        %Mean_Vec(:,i) = mean;
        L = X_train'*X_train/(img_per_person-1);
        [U, S, ~] = svd(L);
        V = X_train*U;
        normv = sqrt(sum(V.^2,1)); 
        V = bsxfun(@rdivide,V,normv);
        Eigen_Vec_3(:,:,i) = V(:,1:3);
    end

    for i = 1:1:num_persons
        for j= 1:1:img_per_person
            TestImg = im2double(reshape(imread(path +"/" +images(((i-1)*img_per_person)+ 2 + j).name),[m*n,1]));
            distance = zeros([num_persons,1]); 
            for k = 1:1:num_persons
                X_test = TestImg - Mean_Vec(:,k);
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
    accuracy = accuracy/N;

%end
toc;