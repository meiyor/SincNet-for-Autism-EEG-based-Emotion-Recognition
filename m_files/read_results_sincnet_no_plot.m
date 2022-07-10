function [err_train,err_test,training_loss,test_loss,err_happy,err_sad,err_angry,err_fear,test_error]=read_results_sincnet_no_plot(path_file,path_extra)
%% path_file: path directory for the initial number of subjects in ISPY - results
%% path_extra: path directory for the extra subjects in ISPY - results
close all;
file=fopen(path_file,'r');
n=1;
%% array initialization
err_train_p=zeros([1,400]);
err_test_p=zeros([1,400]);
training_loss_p=zeros([1,400]);
test_loss_p=zeros([1,400]);
test_error_p=zeros([1,400]);
while(feof(file)~=1)
    data_n=fgets(file);
    test_p(n)=n;
    if length(data_n)==100
        err_train_p(n)=str2double(data_n(39:46));
        err_test_p(n)=str2double(data_n(73:79));
        training_loss_p(n)=str2double(data_n(23:30));
        test_loss_p(n)=str2double(data_n(56:63));
        test_error_p(n)=str2double(data_n(92:99));
    else
        l_b=length(data_n)-100;
        err_train_p(n)=str2double(data_n(39+l_b:46+l_b));
        err_test_p(n)=str2double(data_n(73+l_b:79+l_b));
        training_loss_p(n)=str2double(data_n(23+l_b:30+l_b));
        test_loss_p(n)=str2double(data_n(56+l_b:63+l_b));
        test_error_p(n)=str2double(data_n(92+l_b:99+l_b)); 
    end;
    n=n+1;
end;
err_train=err_train_p(1:n-1);
err_test=err_test_p(1:n-1);
training_loss=training_loss_p(1:n-1);
test_loss=test_loss_p(1:n-1);
test_error=test_error_p(1:n-1);
filep=fopen(path_extra,'r');
n=1;
err_happy_p=zeros([1,400]);
err_sad_p=zeros([1,400]);
err_angry_p=zeros([1,400]);
err_fear_p=zeros([1,400]);
while(feof(filep)~=1)
        data_p=fgets(filep);
        test_p(n)=n;
        if length(data_p)==108
            err_happy_p(n)=str2double(data_p(34:40));
            err_sad_p(n)=str2double(data_p(55:62));
            err_angry_p(n)=str2double(data_p(78:86));
            err_fear_p(n)=str2double(data_p(100:107));
        else
            l_q=length(data_p)-108;
            err_happy_p(n)=str2double(data_p(34+l_q:40+l_q));
            err_sad_p(n)=str2double(data_p(55+l_q:62+l_q));
            err_angry_p(n)=str2double(data_p(78+l_q:86+l_q));
            err_fear_p(n)=str2double(data_p(100+l_q:107+l_q));
        end;
        n=n+1;
end;
err_happy=err_happy_p(1:n-1);
err_sad=err_sad_p(1:n-1);
err_angry=err_angry_p(1:n-1);
err_fear=err_fear_p(1:n-1);
fclose(file);
fclose(filep);
