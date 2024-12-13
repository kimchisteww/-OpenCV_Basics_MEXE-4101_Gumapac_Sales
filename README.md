# Introduction
Contactless biometric fingerprint technology is one where a person can be identified and authenticated without physical contact with a scanning device. It has become essential in machine vision because it is efficient, secure, and easy to use. Frictionless contactless biometrics are more user-friendly. It allows users to authenticate their identity without the discomfort or hygiene concerns associated with touching shared surfaces.



# Abstract

The objective of this activity is to develop a contactless biometric fingerprint. Specifically, it aims to design the algorithm on which the contactless biometric fingerprint is working involved. Next, to develop using programming methods in Python, RoboFlow, and Pytorch for the contactless biometric fingerprint scanning. Lastly, to compare the fingerprints saved through qualitative methods. The Framework used in this activity is RoboFlow using YOLOv8 Model, a Deep Learning Framework to assist developers to develop real-time computer vision projects. The Scope of this project will focus on scanning and saving the fingerprint involved. The activity will not cover the recognizing the fingerprints needed involved.


# Project Methods

1. Using a suitable dataset for identifying the finger involved
   1. Using [RoboFlow FingerTip Only](https://universe.roboflow.com/uidaibiomatch/finger_detect_tip_only), we used the dataset needed to identify the images
      ![image](https://github.com/user-attachments/assets/4eb77421-ff50-4dc2-a458-fca7289bd761)
   2. Then after that we transfer the dataset to our [Google Colab](https://colab.research.google.com/drive/1BT1rFbg-JpeUtq_igNJlOsYNbEsjlKFU#scrollTo=AKVTDpk9nj2R) to train the model
3. Training a model using RoboFlow to identify the finger involved
   1. Stated on the code, we first install ultralytics, roboflow, and download the dataset.
   2. We Train the model using YOLOv8m.pt Model under 20 Epochs in a 640 Image Size
   3. Results are shown in the proceeding graphs
    ```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/20      6.82G      1.225      1.635      1.662          7        640: 100% 88/88 [00:50<00:00,  1.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:03<00:00,  1.55it/s]
                   all        135        128     0.0855      0.586     0.0742     0.0384

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/20      6.75G      1.291      1.221      1.693         11        640: 100% 88/88 [00:48<00:00,  1.83it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.63it/s]
                   all        135        128     0.0214      0.398     0.0136    0.00337

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/20      6.84G      1.262      1.174      1.657          8        640: 100% 88/88 [00:47<00:00,  1.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.76it/s]
                   all        135        128      0.416      0.266      0.236      0.128

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/20      6.85G      1.211      1.065      1.603          4        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  2.46it/s]
                   all        135        128      0.749      0.711      0.763      0.555

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/20      6.83G      1.163      1.023      1.567          6        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.76it/s]
                   all        135        128       0.85      0.812       0.88      0.549

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/20      6.85G      1.133     0.9567      1.527          5        640: 100% 88/88 [00:47<00:00,  1.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.74it/s]
                   all        135        128      0.503      0.805      0.491       0.34

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/20      6.86G        1.1     0.8946      1.501          7        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.74it/s]
                   all        135        128      0.875      0.883      0.942      0.643

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/20      6.85G      1.062     0.8448       1.49          4        640: 100% 88/88 [00:47<00:00,  1.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.56it/s]
                   all        135        128      0.934      0.914      0.967        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/20      6.86G      1.058      0.809      1.456          6        640: 100% 88/88 [00:47<00:00,  1.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.73it/s]
                   all        135        128      0.879      0.891      0.949      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/20      6.85G      1.001     0.7652      1.413          4        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.76it/s]
                   all        135        128      0.928      0.912      0.964      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/20      6.85G     0.8905     0.6032      1.419          3        640: 100% 88/88 [00:48<00:00,  1.83it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  2.12it/s]
                   all        135        128      0.943      0.909      0.974      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/20      6.86G     0.8767     0.5454      1.396          3        640: 100% 88/88 [00:47<00:00,  1.87it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.95it/s]
                   all        135        128      0.937      0.934       0.97       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/20      6.86G     0.8327     0.5319      1.358          3        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.56it/s]
                   all        135        128      0.945      0.939      0.981      0.768

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/20      6.86G     0.8005     0.4752      1.305          3        640: 100% 88/88 [00:47<00:00,  1.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.57it/s]
                   all        135        128      0.928      0.969      0.982      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/20      6.86G     0.8014     0.4609      1.304          3        640: 100% 88/88 [00:47<00:00,  1.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.78it/s]
                   all        135        128      0.926      0.977       0.98      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/20      6.87G     0.7779      0.454      1.284          3        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.73it/s]
                   all        135        128      0.981      0.969      0.985      0.784

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/20      6.86G      0.756     0.4229      1.264          3        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.78it/s]
                   all        135        128      0.976      0.968      0.985      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/20      6.85G     0.7116     0.3946       1.23          3        640: 100% 88/88 [00:47<00:00,  1.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.79it/s]
                   all        135        128      0.986      0.961      0.986      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/20      6.86G     0.6945     0.3764      1.206          3        640: 100% 88/88 [00:47<00:00,  1.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  2.04it/s]
                   all        135        128      0.976      0.969      0.988      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/20      6.87G     0.6765     0.3628      1.188          2        640: 100% 88/88 [00:47<00:00,  1.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:01<00:00,  2.76it/s]
                   all        135        128      0.981      0.969      0.986      0.823
    ```

    And the overall resulted model:
   
    ```
    Model summary (fused): 218 layers, 25,840,918 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.14it/s]
                   all        135        128      0.976      0.969      0.988      0.825
                   tip        128        128      0.976      0.969      0.988      0.825
    Speed: 0.4ms preprocess, 11.3ms inference, 0.0ms loss, 6.5ms postprocess per image
    
    ```
    Confusion Matrix:
    ![image](https://github.com/user-attachments/assets/12851e01-7deb-4088-ab2e-cf46e4602ade)

    Label Results:
    ![image](https://github.com/user-attachments/assets/449f20e8-1517-40fe-acf2-51802bd8fc86)

    Training Results:
    ![image](https://github.com/user-attachments/assets/73ce842e-22b1-4956-b1a5-151b3fc9d1aa)

    Samples of the Selected Train Dataset:
    ![image](https://github.com/user-attachments/assets/580c2373-9d1c-424b-8e3d-1a3ff5bf7532)
    ![image](https://github.com/user-attachments/assets/02f52a6e-07a6-45c4-8347-0c764bdf1c0a)

    Samples of the Valid Label Dataset:
    ![image](https://github.com/user-attachments/assets/67a600fc-25a0-47fa-8b91-4294e9a01a06)
    Samples of the Valid Predicted Dataset:
    ![image](https://github.com/user-attachments/assets/0f23ba64-7492-4246-ba52-e75ea60c6a59)

5. Extracting the model to develop the Contactless Fingerprint Scan
   1. Downloading the best.pt dataset and input on the selected python program
      ![chrome_O0aeVgLSgm](https://github.com/user-attachments/assets/8aaddd18-9df6-414b-b9cc-b4b515850511)

   2. Testing the Trained Model
      Using this code
      ```
      from ultralytics import YOLO

      model = YOLO('FingerDetect.pt')
      
      results = model(source=1, show=True, conf=0.4, save=True)
      
      ```
      We can have a live feedback of the finger detection.

      ![python3 11_H8KDFUwHlx](https://github.com/user-attachments/assets/64c490ff-97cf-48d1-af07-5444c318fa39)

   

7. Developing the Contactless Fingerprint Scan
   1. The following flowchart was designed to integrate the algorithms
      ![Contactless Edge Detection drawio](https://github.com/user-attachments/assets/696a7e2c-1deb-4f87-8a3d-3a659875baf0)
   2. Integrating it into code using VSCode
      ![image](https://github.com/user-attachments/assets/448138b6-4630-4561-aad1-5b4e76ede8a7)
   3. Initial Functional Testing
      
8. Evaluating the Scanned Images Qualitatively

# Conclusion

The summary shows the images of the tests and the improved borders using Sobel Operator Edge Detection.


