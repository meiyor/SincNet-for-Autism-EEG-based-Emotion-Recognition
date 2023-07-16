function read_filters_act_more(subject_code,d_size,sel)
%% run for instance for subject 1730006 read_filters_act_more('1730006',100,1) change second parameter depending on the size of your SincLayer
%% setting up the arrays for data saving
data=zeros([48,399,1000]);
Mdata={};
data_happy=zeros([399,1000]);
data_sad=zeros([399,1000]);
data_angry=zeros([399,1000]);
data_fear=zeros([399,1000]);

n=1;

if sel==1
    for k=1:48 %% trials
        q=1;
        if exist(['filters_act/1730006_filters_act/frequency_ranges_' subject_code '_' num2str(k) '.txt'])
        for i=1:399 %% possible training epochs %% adjust this depending on the size of the filter
            %%ask permission for the filter_vals_folder to Juan Manuel Mayor-Torres
            if exist(['filters_act/1730006_filters_act/filters_vals_' num2str(i) '_' subject_code '_' num2str(k) '_more_filters_n.txt'])
                Mdata{k,q}=dlmread(['filters_act/1730006_filters_act/filters_vals_' num2str(i) '_' subject_code '_' num2str(k) '_more_filters.txt'],',');
                temp_val=zeros([d_size 2000]);
                for p=1:d_size
                    %temp_val(p,:)=freqz(Mdata{k,q}(p,:),1,1000);
                    %% sampling frequency is 500 Hz
                    temp_val(p,:)=fftshift((1/(500*1000)).*(abs(fft(Mdata{k,q}(p,:),2000)).^2));
                end;
                data(k,q,:)=mean(temp_val(:,1001:2000),1);
                q=q+1;
            end;
            i
          end;
          n=n+1;
        end;
        if k==12
            data_happy=squeeze(mean(data(1:12,:,:),1));
        end;
        if k==24
            data_sad=squeeze(mean(data(13:24,:,:),1));
        end;
        if k==36
            data_angry=squeeze(mean(data(25:36,:,:),1));
        end;
        if k==48
            data_fear=squeeze(mean(data(37:n-1,:,:),1));
            save('test_val_filters_response.mat','Mdata','data','data_happy','data_sad','data_angry','data_fear','-v7.3');
        end;
        k
    end;
else
    %% load a template if necessary ask permission to Juan Manuel Mayor-Torres for having access to this file
    load('test_val_filters_1000_ASD_fft_short.mat')
end;
%% reading filter SincNet activation response
v = VideoWriter('Filters_video_responses.avi');
v.FrameRate=16;
open(v);
for i=1:399
    close all;
    plot(linspace(0,250,1000),10*log10(abs(data_happy(i,:))./max(abs(data_happy(i,:)))),'b','LineWidth',2);
    hold on
    plot(linspace(0,250,1000),10*log10(abs(data_sad(i,:))./max(abs(data_sad(i,:)))),'r','LineWidth',2);
    plot(linspace(0,250,1000),10*log10(abs(data_angry(i,:))./max(abs(data_angry(i,:)))),'g','LineWidth',2);
    plot(linspace(0,250,1000),10*log10(abs(data_fear(i,:))./max(abs(data_fear(i,:)))),'k--','LineWidth',2)
    grid on;
    xlabel('Frequency [Hz]');
    ylabel('PSD [dB/Hz]');
    set(gca,'FontSize',17);
    xlim([0 35]);
    title(['ASD Frequency Response epoch=' num2str(i)]);
    legend('Happy','Sad','Angry','Fear');
    ax=gcf;
    frame = getframe(ax);
    frame.cdata=imresize(frame.cdata,[1250,1000]);
    writeVideo(v,frame);
end;
close(v);
