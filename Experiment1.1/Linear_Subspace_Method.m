%%
clear all;
tic;
%% Training on set1 and testing on set2,set3,set4,set5
m = 168; %Image sizes
n = 192;
num_persons = 38; 
path = "../CroppedYale";
%Divides the dataset on the basis of lighting directions
[set1,set2,set3,set4,set5,setcounter] = Create_Subsets(m,n,path);   
Eigen_Vec_3 = zeros([m*n,3,num_persons]);
Mean_Vec = zeros([m*n,num_persons]);
for i = 1:1:num_persons
    X_train = zeros([m*n,setcounter(1,i)]);
    for j = 1:1:setcounter(1,i)
        X_train(:,j) = set1(:,j,i);
    end
    %Note that here we wantedly are not dng mean subtraction as it is decreasing the accuracy for this method
    %mean = sum(X_train,2)/setcounter(1,i); 
    %X_train = X_train - mean;
    %Mean_Vec(:,i) = mean;
    L = X_train'*X_train/(setcounter(1,i)-1);
    [U, S, ~] = svd(L);
    V = X_train*U;
    normv = sqrt(sum(V.^2,1)); 
    V = bsxfun(@rdivide,V,normv);
    Eigen_Vec_3(:,:,i) = V(:,1:3);
end

%% Testing on set2, set3, set4, set5    
accuracy = zeros([4,1]);
count = zeros([4,1]);
for i = 1:1:num_persons
    for j = 1:1:setcounter(2,i)
        TestImg = set2(:,j,i);
        count(1) = count(1) + 1;
        distance = zeros([num_persons,1]); 
        for k = 1:1:num_persons
            X_test = TestImg;
            Test_Coeff = Eigen_Vec_3(:,:,k)'*X_test;
            Reconstruction = Eigen_Vec_3(:,:,k)*Test_Coeff;
            distance(k) = sum((X_test - Reconstruction).^2);                
        end
        [~, indx] = min(distance);
        if indx == i
            accuracy(1) = accuracy(1) + 1;
        end
    end
end

%% Testing on set3
for i = 1:1:num_persons
    for j = 1:1:setcounter(3,i)
        TestImg = set3(:,j,i);
        count(2) = count(2) + 1;
        distance = zeros([num_persons,1]); 
        for k = 1:1:num_persons
            X_test = TestImg;
            Test_Coeff = Eigen_Vec_3(:,:,k)'*X_test;
            Reconstruction = Eigen_Vec_3(:,:,k)*Test_Coeff;
            distance(k) = sum((X_test - Reconstruction).^2);                
        end
        [~, indx] = min(distance);
        if indx == i
            accuracy(2) = accuracy(2) + 1;
        end
    end
end

%% Testing on set4
for i = 1:1:num_persons
    for j = 1:1:setcounter(4,i)
        TestImg = set4(:,j,i);
        count(3) = count(3) + 1;
        distance = zeros([num_persons,1]); 
        for k = 1:1:num_persons
            X_test = TestImg;
            Test_Coeff = Eigen_Vec_3(:,:,k)'*X_test;
            Reconstruction = Eigen_Vec_3(:,:,k)*Test_Coeff;
            distance(k) = sum((X_test - Reconstruction).^2);                
        end
        [~, indx] = min(distance);
        if indx == i
            accuracy(3) = accuracy(3) + 1;
        end
    end
end

%% Testing on set5
for i = 1:1:num_persons
    for j = 1:1:setcounter(5,i)
        TestImg = set5(:,j,i);
        count(4) = count(4) + 1;
        distance = zeros([num_persons,1]); 
        for k = 1:1:num_persons
            X_test = TestImg;
            Test_Coeff = Eigen_Vec_3(:,:,k)'*X_test;
            Reconstruction = Eigen_Vec_3(:,:,k)*Test_Coeff;
            distance(k) = sum((X_test - Reconstruction).^2);                
        end
        [~, indx] = min(distance);
        if indx == i
            accuracy(4) = accuracy(4) + 1;
        end
    end
end
accuracy = (accuracy./count)*100;
disp("Accuracy on the Subset 2 is " + accuracy(1) + "%");
disp("Accuracy on the Subset 3 is " + accuracy(2) + "%");
disp("Accuracy on the Subset 4 is " + accuracy(3) + "%");
disp("Accuracy on the Subset 5 is " + accuracy(4) + "%");
toc;