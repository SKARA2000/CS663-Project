function [accuracy_top30,accuracy_30] = Eigenfaces_method(path,m,n,num_persons,img_per_person,K,flag)
N = num_persons*img_per_person;
images = dir(path);
count = 0;
V_topk = zeros([m*n,K]);
correct_recog_topk = 0;
if(flag==0)
    V_k = zeros([m*n,K]);
    correct_recog_k = 0;
end
for i = 1:1:num_persons
    for j = 1:1:img_per_person
        X_train = zeros([m*n,(img_per_person*num_persons)-1]);
        count=count+1;
        lambda = 0;
        for k=1:(num_persons*img_per_person)
            if(count~=k)
                lambda = lambda+1;
                X_train(:,lambda) = im2double(reshape(imread(path+"/"+images(k+2).name),[(m*n),1]));
            end
        end
        mean = sum(X_train,2)/((img_per_person*num_persons)-1);
        X_train = X_train - mean;
        L = X_train'*X_train/((img_per_person*num_persons)-2);
        [U,S,~] = svd(L);
        V = X_train*U;
        normv = sqrt(sum(V.^2,1));
        V = bsxfun(@rdivide,V,normv);
        V_topk = V(:,1:K);
        if(flag==0)
            V_k = V(:,4:(K+3));
        end
        
        Testimg = im2double(reshape(imread(path+"/"+images(count+2).name),[(m*n),1]));
        Testimg = Testimg - mean;
        Alpha_ik = V_topk'*X_train;
        Alpha_p = V_topk'*Testimg;
        if(flag==0)
            Alpha_ik1 = V_k'*X_train;
            Alpha_p1 = V_k'*Testimg;
        end
        [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
        if(Identity<count)
            Identity = ceil(Identity/img_per_person);
        else
            Identity = ceil((Identity+1)/img_per_person);
        end
        if(Identity == ceil(count/img_per_person))
            correct_recog_topk = correct_recog_topk + 1;
        end
        if(flag==0)
            [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
            if(Identity<count)
                Identity = ceil(Identity/img_per_person);
            else
                Identity = ceil((Identity+1)/img_per_person);
            end
            if(Identity == ceil(count/img_per_person))
                correct_recog_k = correct_recog_k + 1;
            end
        end
    end
end
accuracy_top30 = correct_recog_topk/N;
if(flag==0)
    accuracy_30 = correct_recog_k/N;
else
    accuracy_30 = 0;
end
end