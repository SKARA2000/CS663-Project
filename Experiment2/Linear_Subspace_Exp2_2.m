function [accuracy] = Linear_Subspace_Exp2_2(path,m,n,num_persons,img_per_person)
    N = 15*11;
    images = dir(path);
    accuracy = zeros([img_per_person,1]);
    Eigen_Vec_3 = zeros([m*n,3,num_persons]);

    %% Training and Testing
    for k = 1:1:img_per_person    %Category
        for i = 1:1:num_persons
            X_train = zeros([m*n,img_per_person-1]);
            count = 0;
            for j = 1:1:img_per_person
                if j~=k
                    count = count + 1;
                    X_train(:,count) = im2double(reshape(imread(path +"/" +images(((i-1)*img_per_person) + j + 2).name),[m*n,1]));
                end
            end
            L = X_train'*X_train/(img_per_person-2);
            [U, S, ~] = svd(L);
            V = X_train*U;
            normv = sqrt(sum(V.^2,1)); 
            V = bsxfun(@rdivide,V,normv);
            Eigen_Vec_3(:,:,i) = V(:,1:3);
        end

        %Testing
        for i = 1:1:num_persons
            TestImg = im2double(reshape(imread(path +"/" +images(((i-1)*img_per_person)+ 2 + k).name),[m*n,1]));
            distance = zeros([num_persons,1]);
            for l = 1:1:num_persons
                Test_Coeff = Eigen_Vec_3(:,:,l)'*TestImg;
                Reconstruction = Eigen_Vec_3(:,:,l)*Test_Coeff;
                distance(l) = sum((TestImg - Reconstruction).^2);                
            end
            [~, indx] = min(distance);
            if indx == i
                accuracy(k) = accuracy(k) + 1;
            end
        end    
        accuracy(k) = (accuracy(k)/num_persons)*100;
    end
end