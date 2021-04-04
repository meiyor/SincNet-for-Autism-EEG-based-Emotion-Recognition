# main_EEG_new.py created originally by Mirco Ravanelli
# Modified by Juan Manuel Mayor-Torres
# Cardiff University / University of Trento
# Mila - University of Montreal

# March 2021
# Description: 
# This code performs a EEG-based Emotion Recognition experiments with SincNet.
# We used the data collected from Social Competence Treatment Lab (SCTL) in StonyBrook University, NY, USA

# How to run it:
# python main_EEG_new.py --cfg=cfg/SincNet_EEG.cfg ## use the most realiable tunning parameters on the set

import os
## set the number of available GPU nodes here
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from numpy import genfromtxt
import sys
import numpy as np
from dnn_models_for_sinc import MLP,flip
from dnn_models_for_sinc import SincNet as CNN
from data_io_new import ReadList,read_conf,str_to_bool

## size vector_numbacthes created from the data defined on the config file, 22561=752*30
sig_data_save=np.zeros([num_batches,22561])

def create_batches_rnd(batch_size,data_folder,wlen,lab_dict,fact_amp,i_test_index,epoch_t):
 # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
 #batch_size will be the 48 different iteration per subject for the EEG case
 data_batch=np.zeros([batch_size,wlen])
 lab_batch=np.zeros(batch_size)
 
 sig_train=np.zeros([batch_size,752])
 lab_train=np.zeros(batch_size)  
 ## splitting the subject name string 
 strn1=sys.argv[1].split('_')
 strn2=strn1[4].split('/') 
 
 rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)
 print("creating batches..") 
 ## channels position across the ZCA file
 n=0
 for i in range(batch_size+1):
  #sampling frequency for setting up the random cropping size on the training
  fs=500
  
  if not(i==i_test_index):
        if epoch_t==0:
                training_data_file = genfromtxt(data_folder+'/'+strn1[7]+'/test_data_sub_'+strn1[7]+'_'+str(i+1)+'.csv', delimiter=',', skip_header=0)
                sig_data_save[n,:]=training_data_file
        else:
                training_data_file=sig_data_save[n,:]
        
        ## set all these parameters to read the information from the SCTL ZCA files
        num_cols = training_data_file.shape[0] # Number of data samples
        num_rows = 1 # Length of Data Vector
        total_size_train=(num_cols-1)*num_rows 
        label_train = np.arange(1)
        label_train = label_train.astype('int32')
        data_train = np.arange(total_size_train)
        data_train = data_train.reshape(num_rows, num_cols-1) # 2D Matrix of data points
        data_train = data_train.astype('float32') ## check if thre is some incompatibility
        for k in range(1):
        	label_train[k] = training_data_file[num_cols-1]-1
        	for j in range(num_cols-1):
        		data_train[k][j] = training_data_file[j]

        ## averaging across channels, first reshare the file and average it across the axis 2 with the channels only
        data_train=data_train.reshape((752,30))
        data_train_train=np.mean(data_train,axis=1)
        sig_train[n,:]=data_train
        lab_train_train[n]=label_train[0]
        train_len=sig_train.shape[1]
        train_seg=np.random.randint(train_len-wlen-1)
        len_seg=leg_seg+wlen
        data_batch[n,:]=sig_train[n,train_seg:len_seg]
        lab_batch[n]=lab_train_train[n]

        channels = len(sig_train[n,:].shape)
 
        n=n+1


 input=Variable(torch.from_numpy(data_batch).float().cuda().contiguous())
 labels=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
 return input,labels 


# Reading cfg file you added as input from the command call
options=read_conf()
print(options)

#[data] section
tr_lst=options.tr_lst
te_lst=options.te_lst
pt_file=options.pt_file
class_dict_file=options.lab_dict
print(class_dict_file)
data_folder=options.data_folder+'/'
output_folder=options.output_folder

#[windowing] section
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn] section
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn] section
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class] section
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization] section
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)


# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 

# loss function
cost = nn.NLLLoss()

# Converting context and shift in samples play with this values in milliseconds (ms) taking into account the length of the grand-average ERP representation
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)


# Feature extractor CNN settings
CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
CNN_net.cuda()

## DNN specs loading
DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cuda()

##  DNN2 specs from the classifier-layer softmax..
DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()


if pt_file!='none':
   checkpoint_load = torch.load(pt_file)
   CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
   DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
   DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

# using a learning rate ten times greater for the SincConv and the ConvPool stage in comparison wit the rest of the architecture
optimizer_CNN = optim.Adam(CNN_net.parameters(), lr=lr*10,alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.Adam(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8)
optimizer_DNN2 = optim.Adam(DNN2_net.parameters(), lr=lr,alpha=0.95, eps=1e-8)

strn1=sys.argv[1].split('_')
strn2=strn1[4].split('/')

Batch_dev=100
## saving the frequency ranges if they are needed and take into account for your particular experiment
np.savetxt('frequency_ranges_'+strn1[7]+'_'+sys.argv[2]+'.txt',CNN_net.conv[0].hz)

## Starting the Training section
for epoch in range(N_epochs):
  
  test_flag=0
  CNN_net.train()
  DNN1_net.train()
  DNN2_net.train()
 
  loss_sum=0
  err_sum=0
  err_hsum=0
  err_ssum=0
  err_asum=0
  err_fsum=0 
  print("Training..") ## LOTO crossvalidation modify it
  print(dir(CNN_net))
  ## iterating across the batches you add on the experiments
  for i in range(N_batches):

      ## crete the pytorch variable holders for the EEG collection, the validation should be done for each training process individually
      ## not after the full training for other cross-validation this should be changed in any case. 
      [input,label]=create_batches_rnd(batch_size,data_folder,wlen,lab_dict,0.2,int(sys.argv[2])-1,epoch)
      pout=DNN2_net(DNN1_net(CNN_net(input)))
      pred=torch.max(pout,dim=1)[1]
      loss = cost(pout, label.long())
      err = torch.mean((pred!=label.long()).float())

      err_happy=np.mean((label.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 0.0)[0]]!=(pred.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 0.0)[0]])
      err_sad=np.mean((label.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 1.0)[0]]!=(pred.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 1.0)[0]])
      err_angry=np.mean((label.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 2.0)[0]]!=(pred.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 2.0)[0]])    
      err_fear=np.mean((label.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 3.0)[0]]!=(pred.data).cpu().numpy()[np.where((label.data).cpu().numpy() == 3.0)[0]])    
 
      optimizer_CNN.zero_grad()
      optimizer_DNN1.zero_grad() 
      optimizer_DNN2.zero_grad() 
    
      loss.backward()
      optimizer_CNN.step()
      optimizer_DNN1.step()
      optimizer_DNN2.step()
    
      loss_sum=loss_sum+loss.detach()
      err_sum=err_sum+err.detach()
      err_hsum=err_hsum+err_happy
      err_ssum=err_ssum+err_sad
      err_asum=err_asum+err_angry
      err_fsum=err_fsum+err_fear  
      print(CNN_net.conv[0].hz,np.shape(CNN_net.conv[0].hz),CNN_net.conv[0].filters,np.shape(CNN_net.conv[0].filters),CNN_net.act[0],np.shape(CNN_net.act[0]),'hola') 
  loss_tot=loss_sum/N_batches
  err_tot=err_sum/N_batches
  err_toth=err_hsum/N_batches
  err_tots=err_ssum/N_batches
  err_tota=err_asum/N_batches
  err_totf=err_fsum/N_batches
  print("training epoch #: %i , error_tot: %f, error_happy: %f, error_sad: %f, error_angry: %f, error_fear: %f" % (epoch,err_tot,err_toth,err_tots,err_tota,err_totf))

# Full Validation  new  
  print("Validation..")
  if epoch%N_eval_epoch==0:

     CNN_net.eval()
     DNN1_net.eval()
     DNN2_net.eval()
     test_flag=1 
     loss_sum=0
     err_sum=0
     err_sum_snt=0
   
     with torch.no_grad():  
        for p in range(1):
       
           test_data_file = genfromtxt(data_folder+'/'+strn1[7]+'/test_data_sub_'+strn1[7]+'_'+sys.argv[2]+'.csv', delimiter=',', skip_header=0)
           num_cols_t = test_data_file.shape[0] # Number of data samples
           num_rows_t = 1 # Length of Data Vector
           total_size_test=(num_cols_t-1)*num_rows_t
           label_test = np.arange(1)
           label_test = label_test.astype('int32')
           data_test = np.arange(total_size_test)
           data_test = data_test.reshape(num_rows_t, num_cols_t-1) # 2D Matrix of data points
           data_test= data_test.astype('float32') ## check if thre is some incompatibility
           for k in range(1):
                label_test[k] = test_data_file[num_cols_t-1]-1
                for j in range(num_cols_t-1):
                    data_test[k][j] = test_data_file[j]
 
           ## averaging across channels
           data_test=data_test.reshape((752,30))
           data_test_t=np.mean(data_test,axis=1)
           signal=torch.from_numpy(data_test_t).float().cuda().contiguous()
           lab_batch=label_test[0]

           # split signals into random chunks
     
           beg_samp=0
           end_samp=wlen

           N_fr=int((signal.shape[0]-wlen)/(wshift))


           sig_arr=torch.zeros([100,wlen]).float().cuda().contiguous()
           lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
           pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
     
           count_fr=0
           count_fr_tot=0

           ## create the random array
           while end_samp<signal.shape[0]:
                 sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                 beg_samp=beg_samp+wshift
                 end_samp=beg_samp+wlen
                 count_fr=count_fr+1
                 count_fr_tot=count_fr_tot+1
                 if count_fr==Batch_dev:
                    inp=Variable(sig_arr)
                    pout[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
                    count_fr=0
                    sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()

           if count_fr>0:
              inp=Variable(sig_arr[0:count_fr])
              pout[count_fr_tot-count_fr:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))

           pred=torch.max(pout,dim=1)[1]
           loss = cost(pout, lab.long())
           err = torch.mean((pred!=lab.long()).float())
    
           [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
           err_sum_snt=err_sum_snt+(best_class!=lab[0]).float()
           err_val=(pred!=lab.long()).float()

           loss_sum=loss_sum+loss.detach()
           err_sum=err_sum+err.detach()
    
        err_tot_dev_snt=err_sum_snt
        loss_tot_dev=loss_sum
        err_tot_dev=err_sum

        ## here we save the filters learnt for each cross-val iteration
        np.savetxt('filters_vals_'+str(epoch)+'_'+strn1[7]+'_'+sys.argv[2]+'.txt',np.squeeze(CNN_net.conv[0].filters.cpu().numpy()),delimiter=',')
        with open('folder_output'+'/res_error_'+strn1[7]+'_'+sys.argv[2]+'.csv','a') as error_file:
            for err_dat in err_b:
                error_file.write(str(err_dat.float())+",")
            error_file.write("%f,%f \n" % (err,(best_class!=lab[0]).float()))

     ## saving metrics data 
     print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))
     print("train-epoch %i, error_happy: %f, error_sad: %f, error_angry: %f, error_fear: %f" % (epoch,err_toth,err_tots,err_tota,err_totf))

     with open(output_folder+"/res_evaluation_"+strn1[7]+'_'+sys.argv[2]+".csv", "a") as res_file:
        res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))

     checkpoint={'CNN_model_par': CNN_net.state_dict(),
               'DNN1_model_par': DNN1_net.state_dict(),
               'DNN2_model_par': DNN2_net.state_dict(),
               }
     torch.save(checkpoint,output_folder+'/model_file.pkl')

  else:
     print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))


