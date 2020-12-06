%%
clear all;
tic;
%% Training on set1,set5 and testing on set2,set3,set4
m = 168;
n = 192;
num_persons = 38;
path = "../CroppedYale";
[set1,set2,set3,set4,set5,setcounter] = Create_Subsets(m,n,path);
accuracy = zeros([3,1]);
count = zeros([3,1]);
Eigen_Vec_3 = zeros([m*n,3,num_persons]);
Mean_Vec = zeros([m*n,num_persons]);
for i = 1:1:num_persons
    X_train = zeros([m*n,setcounter(1,i)+setcounter(5,i)]);
    for j = 1:1:setcounter(1,i)
        X_train(:,j) = set1(:,j,i);
    end
    for j = 1:1:setcounter(5,i)
        X_train(:,j+setcounter(1,i)) = set5(:,j,i);
    end
    %Note that here we wantedly are not dng mean subtraction as it is decreasing the accuracy for this method
    %mean = sum(X_train,2)/(setcounter(1,i) + setcounter(5,i));
    %X_train = X_train - mean;
    %Mean_Vec(:,i) = mean;
    L = X_train'*X_train/(setcounter(1,i) + setcounter(5,i) - 1);
    [U, S, ~] = svd(L);
    V = X_train*U;
    normv = sqrt(sum(V.^2,1)); 
    V = bsxfun(@rdivide,V,normv);
    Eigen_Vec_3(:,:,i) = V(:,1:3);
end

%% Testing on set2
for i = 1:1:num_persons
    for j = 1:1:setcounter(2,i)
        TestImg = set2(:,j,i);
        count(1) = count(1) + 1;
        distance = zeros([num_persons,1]); 
        for k = 1:1:num_persons
            X_test = TestImg - Mean_Vec(:,k);
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
            X_test = TestImg - Mean_Vec(:,k);
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
            X_test = TestImg - Mean_Vec(:,k);
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
accuracy = (accuracy./count)*100;
disp("Accuracy on the Subset 2 is " + accuracy(1) + "%");
disp("Accuracy on the Subset 3 is " + accuracy(2) + "%");
disp("Accuracy on the Subset 4 is " + accuracy(3) + "%");
toc;
%end