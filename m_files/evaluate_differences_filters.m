function [p,tbl]=evaluate_differences_filters(path_TD,path_ASD,pos_freq,sel)
%% pos_freq: is a 2D vector with the positions on the frequency response you want to analyze
%% sel: selector for evaluate anova 1-way or anova n-way
D_TD=dir(path_TD);
D_ASD=dir(path_ASD);
t=linspace(0,250,1000) %% the frequency response
pos1=min(find(t>=pos_freq(1)));
pos2=max(find(t<=pos_freq(2))); 
%% first read all the data for all the TDs
for i=3:length(D_TD)
    close all;
    d_TD=load([path_TD '/' D_TD(i).name])
    %% set the d_td for each subject in the folder that contains the folder of TD filters 
    temp=mean([d_TD.data_happy(:,pos1:pos2) ; d_TD.data_sad(:,pos1:pos2) ; d_TD.data_angry(:,pos1:pos2) ; d_TD.data_fear(:,pos1:pos2)]),1); 
    d_td(i)=mean(10*log10(abs(temp)./max(abs(temp))));
end;
%% now read all the data for all the ASDs
for i=3:length(D_ASD)
    close all;
    d_ASD=load([path_ASD '/' D_ASD(i).name])
    %% set the d_td for each subject in the folder that contains the folder of TD filters 
    temp=mean([d_ASD.data_happy(:,pos1:pos2) ; d_ASD.data_sad(:,pos1:pos2) ; d_TD.data_angry(:,pos1:pos2) ; d_TD.data_fear(:,pos1:pos2)],1); 
    d_asd(i)=mean(10*log10(abs(temp)./max(abs(temp))));
end;
%% now having all the filters averaged we can implement anova to compare the differences between the time you want to compare 
if sel == 0 %% select anova1
   [p,tbl,stats]=anova1([d_td d_asd]',{[ones([1 length(d_td)]) 2*ones([1 length(d_asd)])]});
else
   [p,tbl,stats]=anovan([d_td d_asd]',{[ones([1 length(d_td)]) 2*ones([1 length(d_asd)])]},'model','full');
end;   
%% return the p-values and the stats coming from each anova evaluation
