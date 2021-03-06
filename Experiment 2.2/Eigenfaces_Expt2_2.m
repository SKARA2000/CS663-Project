function [accuracy,accuracy1] = Eigenfaces_Expt2_2(path,m,n,num_persons,img_per_person,K,flag)
N = num_persons*img_per_person;
images = dir(path);
V_topk = zeros([m*n,K]);
accuracy = zeros(img_per_person,1);
if(flag==0)
    V_k = zeros([m*n,K]);
    accuracy1 = zeros(img_per_person,1);
end
for k = 1:1:img_per_person
    count = 0;
    X_train = zeros([m*n,(img_per_person-1)*num_persons]);
    for i = 1:1:num_persons
        for j = 1:1:img_per_person
            if j~=k
                count = count + 1;
                X_train(:,count) = im2double(reshape(imread(path +"/" +images(((i-1)*img_per_person) + j + 2).name),[m*n,1]));
            end
        end
    end
    mean = sum(X_train,2)/(setcounter(1,i) + setcounter(5,i));
    X_train = X_train - mean;
    L = X_train'*X_train/(((img_per_person-1)*num_persons)-1);
    [U, S, ~] = svd(L);
    V = X_train*U;
    normv = sqrt(sum(V.^2,1)); 
    V = bsxfun(@rdivide,V,normv);
    V_topk = V(:,1:K);
    if(flag==0)
        V_k = V(:,4:(K+3));
        correct_recog_k = 0;
    end

    correct_recog_topk = 0;
    for i = 1:1:num_persons
        Testimg = im2double(reshape(imread(path +"/" +images(((i-1)*img_per_person)+ 2 + k).name),[m*n,1]));
        Testimg = Testimg - mean;
        Alpha_ik = V_topk'*X_train;
        Alpha_p = V_topk'*Testimg;
        if(flag==0)
            Alpha_ik1 = V_k'*X_train;
            Alpha_p1 = V_k'*Testimg;
        end
        [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
        Identity = ceil((Identity+1)/(img_per_person-1));
        if(Identity == i)
            correct_recog_topk = correct_recog_topk + 1;
        end
        if(flag==0)
            [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
            Identity = ceil((Identity+1)/(img_per_person-1));
            if(Identity == i)
                correct_recog_k = correct_recog_k + 1;
            end
        end
    end    
    accuracy(k) = correct_recog_topk/num_persons;
    if(flag==0)
        accuracy1(k) = correct_recog_k/num_persons;
    else
        accuracy1(k) = 0;
    end
end