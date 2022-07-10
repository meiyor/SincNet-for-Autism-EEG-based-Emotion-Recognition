function [test_err_sal,performance,err_av_sal,loss_av_sal,err_av,err_happy,err_sad,err_angry,err_fear,Cn]=compile_res_folder(path)
%%% define the path of your results before running this function
Adir=dir(path);
pdfolder_d=strsplit(path,'/');
pdfolder=strsplit(pdfolder_d{end},'\');
%%set the initials counters
n=1;
q=1;

for p=1:48 %% running along trials
        [err_train{p},err_test{p},training_loss{p},test_loss{p},err_happy{p},err_sad{p},err_angry{p},err_fear{p},test_error{p}]=read_results_sincnet_no_plot([path '/res_time_av_test_data_sub_' pdfolder{end} '_{' num2str(p) '}_err_te.csv'],[path '/res_time_av_test_data_sub_' pdfolder{end} '_{' num2str(p) '}_train_epoch.csv']);
        %[err_train{p},err_test{p},training_loss{p},test_loss{p},err_happy{p},err_sad{p},err_angry{p},err_fear{p},test_error{p}]=read_results_sincnet_no_plot([path '\res_time_av_test_data_sub_' pdfolder{end} '_{' num2str(p) '}_err_te.csv'],[path '\res_time_av_test_data_sub_' pdfolder{end} '_{' num2str(p) '}_train_epoch.csv']);
        %[err_train{p},err_test{p},training_loss{p},test_loss{p},err_happy{p},err_sad{p},err_angry{p},err_fear{p},test_error{p}]=read_results_sincnet_no_plot([path '\res_time_channel_14_test_data_sub_' pdfolder{end} '_{' num2str(p) '}_err_te.csv'],[path '\res_time_channel_14_test_data_sub_' pdfolder{end} '_{' num2str(p) '}_train_epoch.csv']);
        if isfile([path '/res_error_' pdfolder{end} '_' num2str(p) '.csv'])
            index_n(q)=p;
            [data_res{p}]=read_error_location([path '/res_error_' pdfolder{end} '_' num2str(p) '.csv']);
            ss=size(data_res{p});
            if ss(1)<=370
                if ss(1) == 200
                         data_res{p}=[data_res{p}(1:200,:) ; zeros([200,ss(2)])];  
                else
                         data_res{p}=[data_res{p}(1:ss(1),:) ; zeros([400-ss(1),ss(2)])];
                end; 
            else
                if ss(1) == 400
                    data_res{p}=data_res{p}(1:400,:);
                elseif ss(1)<=400
                    data_res{p}=[data_res{p}(1:ss(1),:) ; zeros([400-ss(1),ss(2)])];
                else
                    data_res{p}=data_res{p}(1:400,:);
                end;
            end;
            if ss(2)~=202
                data_res{p}=data_res{p}(:,round(linspace(1,ss(2),202)));
            end;
            q=q+1;
        else
            index_n(q)=p;
            data_res{p}=data_res{p-1};
        end;
        if length(err_test{p})>=200
            err_test{p}=err_test{p}(1:200);
            test_loss{p}=test_loss{p}(1:200);
            test_error{p}=test_error{p}(1:200);
        end;
        if p~=1 && length(err_test{p})~=370 && length(err_test{p})~=200 
                err_test{p}=err_test{p-1};
                test_loss{p}=test_loss{p-1};
                test_error{p}=test_error{p-1};
        end;
        p
end;
err_sal=cell2mat(err_test');
test_err_sal=cell2mat(test_error');
performance=1-round(mean(test_err_sal(:,:),2));
loss_sal=cell2mat(test_loss');
err_av_sal=mean(err_sal,1);
loss_av_sal=mean(loss_sal,1);

%% calculate averages for the errors across
err_happy=zeros([size(data_res{1},1) size(data_res{1},2)]);
err_sad=zeros([size(data_res{1},1) size(data_res{1},2)]);
err_angry=zeros([size(data_res{1},1) size(data_res{1},2)]);
err_fear=zeros([size(data_res{1},1) size(data_res{1},2)]);
err_av=zeros([size(data_res{1},1) size(data_res{1},2)]);
k=1;
n=[0 0 0 0];
for i=1:48
    if any(index_n==i)
        if i>=1 && i<=12
            err_happy=err_happy+data_res{i};%(1:size(data_res{1},1),1:size(data_res{1},2));
            n(1)=n(1)+1;
        elseif i>=13 && i<26
            err_sad=err_sad+data_res{i};%(1:size(data_res{1},1),1:size(data_res{1},2));
            n(2)=n(2)+1;
         elseif i>=26 && i<38
            err_angry=err_angry+data_res{i};%(1:size(data_res{1},1),1:size(data_res{1},2));
            n(3)=n(3)+1;
        else
            err_fear=err_fear+data_res{i};%(1:size(data_res{1},1),1:size(data_res{1},2));
            n(4)=n(4)+1;
        end;
        err_av=err_av+data_res{i};%(1:size(data_res{1},1),1:size(data_res{1},2));
        k=k+1;
    end;
    i
end;
err_happy=err_happy./n(1);
err_sad=err_sad./n(2);
err_angry=err_angry./n(3);
err_fear=err_angry./n(4);
err_av=err_av./(k-1);
Cn=process_conf_matrix(1-performance,err_happy,err_sad,err_angry,err_fear);
