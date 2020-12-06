%function [accuracy_top30,accuracy_30] = Eigenfaces_method(path,m,n,num_persons,img_per_person,K,flag)
m = 168;
n = 192;
num_person = 38;
path = "../CroppedYale";
K=140;
flag=0;
[set1,set2,set3,set4,set5,setcounter] = Create_Subsets(m,n,path);
V_topk = zeros([m*n,K]);
%correct_recog_topk = zeros(3);
accuracy=zeros([1,4]);
if(flag==0)
    V_k = zeros([m*n,K]);
    correct_recog_topk = zeros(4);
end
a=0;
b=0;
d=0;
e=0;
sumcolumn = zeros([1,5]);
sumcolumn=sum(setcounter');
x=zeros([1,38]);
X_train= zeros([m*n,sumcolumn(1)]);
c=0;
for i = 1:1:num_person
    for j = 1:1:setcounter(1,i)
        X_train(:,j+c) = set1(:,j,i);
    end
    c=c+setcounter(1,i);
    x(i)=c;
end
   
mean = sum(X_train,2)/(sumcolumn(1));
X_train = X_train - mean;
L = X_train'*X_train/(sumcolumn(1)-1);
[U,S,~] = svd(L);
V = X_train*U;
normv = sqrt(sum(V.^2,1));
V = bsxfun(@rdivide,V,normv);
V_topk = V(:,1:K);
if(flag==0)
    V_k = V(:,4:(K+3));
end


for i = 1:1:num_person 
    for j = 1:1:setcounter(2,i)
            Testimg = set2(:,j,i);
            Testimg=Testimg-mean;
            Alpha_ik = V_topk'*X_train;
            Alpha_p = V_topk'*Testimg;
            if(flag==0)
                Alpha_ik1 = V_k'*X_train;
                Alpha_p1 = V_k'*Testimg;
            end
            [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
            index=1;
             while x(index)<Identity
                index=index+1;
            end
            if(index==i)
                a=a+1;
            end             
    end
    
    
    for j = 1:1:setcounter(3,i)
            Testimg = set3(:,j,i);
            Testimg=Testimg-mean;
            Alpha_ik = V_topk'*X_train;
            Alpha_p = V_topk'*Testimg;
            if(flag==0)
                 Alpha_ik1 = V_k'*X_train;
                 Alpha_p1 = V_k'*Testimg;
            end
            [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
            index=1;
            while x(index)<Identity
                index=index+1;
            end
            if(index==i)
                b=b+1;
            end             
    end
    
    
    for j = 1:1:setcounter(4,i)
            Testimg = set4(:,j,i);
            Testimg=Testimg-mean;
            Alpha_ik = V_topk'*X_train;
            Alpha_p = V_topk'*Testimg;
            if(flag==0)
                Alpha_ik1 = V_k'*X_train;
                Alpha_p1 = V_k'*Testimg;
            end
            [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
            index=1;
            while x(index)<Identity
                index=index+1;
            end
            if(index==i)
               d=d+1;
            end             
    end
    
    for j = 1:1:setcounter(5,i)
            Testimg = set5(:,j,i);
            Testimg=Testimg-mean;
            Alpha_ik = V_topk'*X_train;
            Alpha_p = V_topk'*Testimg;
            if(flag==0)
                Alpha_ik1 = V_k'*X_train;
                Alpha_p1 = V_k'*Testimg;
            end
            [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
            index=1;
            while x(index)<Identity
                index=index+1;
            end
            if(index==i)
               e=e+1;
            end             
    end
    
end
accuracy(1)=a/sumcolumn(2);
accuracy(2)=b/sumcolumn(3);
accuracy(3)=d/sumcolumn(4);
accuracy(4)=e/sumcolumn(5);
disp(accuracy);
accuracy_top30 = accuracy;
if(flag==0)
    accuracy_30 = accuracy;
end
%end