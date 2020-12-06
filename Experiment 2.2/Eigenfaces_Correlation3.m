%function [accuracy_top30,accuracy_30] = Eigenfaces_method(path,m,n,num_persons,K,flag)
m = 168;
n = 192;
num_person = 38;
path = "../CroppedYale";
K=30;
flag=0;
[set1,set2,set3,set4,set5,setcounter] = Create_Subsets(m,n,path);
sumcolumn = zeros([1,5]);
sumcolumn=sum(setcounter');
accuracy = zeros([3,1]);
count2 = zeros([3,1]);
count=0;
Mean_Vec = zeros([m*n,num_person])
V_topk = zeros([m*n,K]);
correct_recog_topk = zeros([3,1]);
if(flag==0)
    V_k = zeros([m*n,K]);
    correct_recog_k = 0;
end

for i = 1:1:num_person
    X_train = zeros([m*n,setcounter(1,i)+setcounter(5,i)]);
    for j = 1:1:setcounter(1,i)
        X_train(:,j) = set1(:,j,i);
    end
    for j = 1:1:setcounter(5,i)
        X_train(:,j+setcounter(1,i)) = set5(:,j,i);
    end
    mean = sum(X_train,2)/((setcounter(1,i) + setcounter(5,i))-1);
    X_train = X_train - mean;
    Mean_Vec(:,i) = mean;
    L = X_train'*X_train/((setcounter(1,i) + setcounter(5,i))-2);
    [U,S,~] = svd(L);
    V = X_train*U;
    normv = sqrt(sum(V.^2,1));
    V = bsxfun(@rdivide,V,normv);
    V_topk = V(:,1:K);
    if(flag==0)
        V_k = V(:,4:(K+3));
    end
    
    
    for j = 1:1:setcounter(2,i)
        Testimg = set2(:,j,i);
        Alpha_ik = V_topk'*X_train;
        Alpha_p = V_topk'*Testimg;
        if(flag==0)
            Alpha_ik1 = V_k'*X_train;
            Alpha_p1 = V_k'*Testimg;
        end
        [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
        if(Identity==i)
            correct_recog_topk(1)=correct_recog_topk(1)+1;
        end
    end
    
    
    for j = 1:1:setcounter(3,i)
        Testimg = set3(:,j,i);
        Alpha_ik = V_topk'*X_train;
        Alpha_p = V_topk'*Testimg;
        if(flag==0)
            Alpha_ik1 = V_k'*X_train;
            Alpha_p1 = V_k'*Testimg;
        end
        [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
        if(Identity==i)
            correct_recog_topk(2)=correct_recog_topk(2)+1;
        end
    end
    
    for j = 1:1:setcounter(4,i)
        Testimg = set4(:,j,i);
        Alpha_ik = V_topk'*X_train;
        Alpha_p = V_topk'*Testimg;
        if(flag==0)
            Alpha_ik1 = V_k'*X_train;
            Alpha_p1 = V_k'*Testimg;
        end
        [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
        if(Identity==i)
            correct_recog_topk(3)=correct_recog_topk(3)+1;
        end
    end
    
end
 correct_recog_topk(3)=correct_recog_topk(3)/sumcolumn(4);
 correct_recog_topk(3)=correct_recog_topk(2)/sumcolumn(3);
 correct_recog_topk(3)=correct_recog_topk(1)/sumcolumn(2);
accuracy_top30 = correct_recog_topk;
if(flag==0)
    accuracy_30 = correct_recog_topk;
else
    accuracy_30 = 0;
end