function C=process_conf_matrix(indicator,err_h,err_s,err_a,err_f)
## contruct the confusion matrix and its metric given the LOTO errors
C=zeros(4,4);
err=[mean(mean(err_h)),mean(mean(err_s)),mean(mean(err_a)),mean(mean(err_f))];
for k=1:length(indicator)
    if k>=1 && k<=12
            pos=1;
        elseif k>=13 && k<25
            pos=2;
         elseif k>=26 && k<38
            pos=3;
        else
            pos=4;
        end;
    if indicator(k)==0
        C(pos,pos)=C(pos,pos)+1;
    else
        pos_err=find(err==min(err));
        v_s=sort(err);
        if pos==pos_err
            pos_err=find(err==v_s(2));
            C(pos,pos_err)=C(pos,pos_err)+1;
        else
            C(pos,pos_err)=C(pos,pos_err)+1;
        end;
    end;
end;
