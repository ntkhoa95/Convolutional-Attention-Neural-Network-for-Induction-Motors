# Effective Fault Diagnosis Based on Wavelet and Convolutional Attention Neural Network for Induction Motors

Implementation of [Effective Fault Diagnosis Based on Wavelet and Convolutional Attention Neural Network for Induction Motors (IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT)](https://ieeexplore.ieee.org/document/9666871)

## Introduction
Induction motors are important equipment in modern industry. However, the occurrence of fatigue failure following an extended period of operation invariably results in a catastrophic failure. As a result, monitoring and diagnosing induction motors is critical to avoiding unplanned shutdowns caused by premature failures. This article aims to develop an effective method for motor fault detection using time–frequency contents of vibration signals and an attention-based convolutional neural network model. First, the vibration signals are collected and labeled into five different categories: normal condition, outer ring fault, inner ring fault, misalignment condition, and broken rotor bar. Then, using the Morlet function, continuous wavelet transform (CWT) converts the vibratory time-series signals to the scalogram feature images. The time–frequency feature images are created after downsampling and converting the measured vibration signals to the frequency domain. These images are then resized and fed into the proposed convolutional attention neural network (CANN) to identify various induction motor failures. The experimental results demonstrate that the suggested model canprovide an excellent diagnosis accuracy of 99.43%, significantlybetter than the state-of-the-art deep learning approaches forfault diagnosis. Moreover, the developed model’s robustness is validated against adversarial attacks based on the fast gradient sign method (FGSM) by including white Gaussian noise.


## Citation
If you use CANN in an academic work, please cite:

@ARTICLE{9666871,
  author={Tran, Minh-Quang and Liu, Meng-Kun and Tran, Quoc-Viet and Nguyen, Toan-Khoa},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Effective Fault Diagnosis Based on Wavelet and Convolutional Attention Neural Network for Induction Motors}, 
  year={2022},
  volume={71},
  number={},
  pages={1-13},
  doi={10.1109/TIM.2021.3139706}}
