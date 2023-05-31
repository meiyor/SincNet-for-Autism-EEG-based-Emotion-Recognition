# SincNet-for-Autism-EEG-based-Emotion-Recognition

This repository is inspired on the original implementation of [SincNet](https://github.com/mravanelli/SincNet). Here we modified and adapted the original SincNet code to evaluate the performance of a SincNet-based architecture for EEG-based emotion recognition. For this evaluation we utilized EEG data collected on the [Social Competence and Treatment Lab (SCTL)](https://www.lernerlab.com/) from StonyBrook University, NY, USA. The EEG data were collected from 40 individual diagnosed with Autism-Spectrum-Disorder (ASD) and 48 typically-developed (TD) or non-ASD participants. For any data request (e.g., behavioral, EEG) or analysis, pleased send an email to the corresponding contributors at juan.mayortorres@unitn.it, mirco.ravanelli@gmail.com, or matthew.lerner@stonybrook.edu and be patient during the request. The original SincNet paper describing the architecture and the corresponding test on speech and speaker recognition is here [Ravanelli & Bengio 2018](https://arxiv.org/abs/1812.05920).

If you want to use this pipeline and/or request the dataset associated to this repository please cite our paper [Mayor-Torres, J. M., Ravanelli, M., Medina-DeVilliers, S. E., Lerner, M. D., & Riccardi, G. (2021). Interpretable SincNet-based Deep Learning for Emotion Recognition from EEG brain activity. 2021 43rd Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)](https://ieeexplore.ieee.org/abstract/document/9630427)

The proposed SincNet-based architecture for EEG-based emotion recognition is described on the following Figure:
<img src="https://github.com/meiyor/SincNet-for-Autism-EEG-based-Emotion-Recognition/blob/main/pipeline_sincnet_alone.jpg" width="900" height="310">

This system is composed of a SincConv layer, three standard conv-pool blocks such as Conv1, Conv2, and Conv3, and a fully-connected DNN1 coupled and connected to a softmax classifier. The three 2D convolutional blocks are based on 32, 64, and 128 channels with kernel sizes of (100 x 10), (20 x 5), (5 x 2), respectively. Max-pooling used kernel sizes of (10 x 5), (5 x 2), (2 x 2). All the DNN units were ReLUs.

After you downloaded the data from the request you can install the package requirements on the  [SincNet repository](https://github.com/mravanelli/SincNet) and run the leave-one-trial-out (LOTO) cross-validation using the following command: 

```python
python main_SincNet_EEG_model.py "/folder/with_TD_or_ASD_ZCA_inputs/"
```
Please be careful where you set up your data folder containing the 40, 48 or any amount of participant data you want to use in your experiment. This process can take a lot of execution time and memory. For a P100 GPU the full simulation on the 88 participants took approximately four complete days. Running the process the results will be saved on temporary files with the name "res_evaluation_subjectID_trialnumber.csv" containing the loss and the accuracy average metrics, and files with the name "res_error_subjectID_trialnumber.csv" contains the binary representation of **hits** when the test trial label matches with the prediction and a **miss** otherwise.

The performance comparison between the human (participant) performance, a CNN baseline described [here](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals), and the SincNet-based system is plotted using a barplot in the following Figure:

<img src="https://github.com/meiyor/SincNet-for-Autism-EEG-based-Emotion-Recognition/blob/main/comp_sinc_paper_EMBC.jpg" width="700" height="330">

An important feature of our proposed SincNet-based system is to learn the frequency responses from TD and ASD participants using a by-design interpretability resource. This consists in an application of a Fast-Fourier-Transform on each filters' impulse-response learned from the SincConv layer. The following .gif loop shows the Power-Spectral-Density (PSD) for TD (blue) and ASD (red) individuals across all the 400 training loops we used on this experiment. This information is learned in unsupervised (implied) way without adding any diagnosis label.

<img src="https://github.com/meiyor/SincNet-for-Autism-EEG-based-Emotion-Recognition/blob/main/output_more_summary_TD_ASD.gif" width="600" height="310">

After 400 training iterations we obtained significant differences on high-α (9-13 Hz) **F(1,87)=3.331,p=0.000267** and β (13-30 Hz) **F(1,87)=2.05,p=0.00102** bands after Bonferroni-Holm correction. This significant difference are consistent with previous EEG studies which includes data from individuals with ASD [Pineda et., al 2012](https://www.sciencedirect.com/science/article/pii/S0306987712004082?casa_token=jF6BBvZsuFgAAAAA:cZuNKDgpQg1lv5Y2vmoKyONX2ifYx9-48EmbIXOZ1YT_OTGgsI0iWq130jQ6A9w8JMZP-RYOhQ) and [Friederich et., al 2015](https://link.springer.com/article/10.1007/s10803-015-2523-5).

To process all the results for filter activation, metrics and results wrapping use all the code released in the [m_files](https://github.com/meiyor/SincNet-for-Autism-EEG-based-Emotion-Recognition/tree/main/m_files) directory. 
