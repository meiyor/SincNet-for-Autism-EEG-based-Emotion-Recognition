function [data_res]=read_error_location(path_file)
file=fopen(path_file,'r');
q=1;
while(feof(file)~=1)
    data_n=fgets(file);
    data_k=strsplit(data_n,',');
    p=1;
    for n=1:length(data_k)
        if data_k{n}(1)=='t'
            data_res(q,p)=str2num(data_k{n}(8));
            p=p+1;
        end;
    end;
    q=q+1;
end;
data_res=1-data_res;
fclose(file); 
