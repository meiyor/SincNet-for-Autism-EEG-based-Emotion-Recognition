import os
import sys
import subprocess
"""
main function for running the outter loop of the Leave-One-Trial-Out (LOTO) cross-validation. Please add the full path of the ZCA/EEG images in sys.argv[1]
"""
for root, dirnames, filenames in os.walk(str(sys.argv[1])):
    # these for loops define the leave-one-trial-out (LOTO) cross-validation for TD or ASD folders
    # please define this in str(sys.argv[1])
    for name in sorted(dirnames):
        file_name = sorted(os.listdir(''.join([str(sys.argv[1]), '/', name])))
        for files in file_name:
            print(''.join([str(sys.argv[1]), '/', name, '/', files]))
            if os.path.isfile(''.join([str(sys.argv[1]), '/', name, '/', files])):
                # before run each subjects run counter_divide_loto.sh and creates all the temporary training and test files,
                # ALL THE TEMPORARY FILES SHOULD BE CREATED A PRIORI
                # define the result folder per subject ONLY USE THIS TO TES$
                if "train" in files and os.path.isdir(''.join([str(sys.argv[1]), '/', name, '/Results_Vals'])) == 0:
                    str1 = ''.join([str(sys.argv[1]), '/', name, '/', files])
                    str_test = "test"+files[5:]
                    print(str_test)
                    str2 = ''.join(
                        [str(sys.argv[1]), '/', name, '/', str_test])
                    strp = str_test.split('_')
                    strp = strp[4].split('.')
                    print(strp[0])
                    # please be sure your cfg file is added here to do the correct evaluation
                    subprocess.call('python -u main_EEG_SincNet.py "%s" "%s" --cfg=cfg/SincNet_Config_vals.cfg 
                                                                               | tee folder_output/res_time_"%s".txt ' % (str2, strp[0], str_test), shell=True)
