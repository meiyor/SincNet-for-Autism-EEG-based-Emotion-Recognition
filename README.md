# SincNet-for-Autism-EEG-based-Emotion-Recognition

This repository is inspired on the original implementation of [SincNet](https://github.com/mravanelli/SincNet). Here we modified and adapted the original SincNet code to evaluate the performance of a SincNet-based architecture for EEG-based emotion recognition. For this evaluation we utilized EEG data collected on the [Social Competence and Treatment Lab (SCTL)](https://www.lernerlab.com/) from StonyBrook University, NY, USA. The EEG data were collected from 40 individual diagnosed with Autism-Spectrum-Disorder (ASD) and 48 typically-developed (TD) or non-ASD participants. For any data request (e.g., behavioral, EEG) or anaylysis, pleased send an email to the corresponding contributors at juan.mayortorres@unitn.it, mirco.ravanelli@gmail.com, or matthew.lerner@stonybrook.edu.The original SincNet paper describing the architecture and the corresponding test on speech and speaker recognition is here [Ravanelli & Bengio 2018](https://arxiv.org/abs/1812.05920).

The proposed SincNet-based architecture for EEG-based emotion recognition is described on the following Figure:
<img src="https://github.com/meiyor/SincNet-for-Autism-EEG-based-Emotion-Recognition/blob/main/pipeline_sincnet_alone.jpg" width="900" height="310">

This system is composed of a SincConv layer, three standard conv-pool blocks such as Conv1, Conv2, and Conv3, and a fully-connected DNN1 coupled and connected to a softmax classifier. The three 2D convolutional blocks are based on 32, 64, and 128 channels with kernel sizes of (100 x 10), (20 x 5), (5 x 2), respectively. Max-pooling used kernel sizes of (10 x 5), (5 x 2), (2 x 2). All the DNN units were ReLUs.

After you downloaded the data from the request you can install the package requirements on the  [SincNet](https://github.com/mravanelli/SincNet) and run the leave-one-trial-out (LOTO) cross-validation using the following command: 
