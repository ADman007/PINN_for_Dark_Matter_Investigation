# <p align="center"> ML4SCI_26 </p>
### DeepLense - Physics Guided Machine Learning on Real Lensing Images
![image](https://github.com/user-attachments/assets/4c2f33d4-8ba2-46bc-83b9-eef6364b0cc4)
---
## Common Test I. Multi-Class Classification

### Dataset and Classes  
The dataset consists of three types of lensing images:  

- **No Substructure**  
- **Sphere Substructure**  
- **Vortex Substructure**  

### Preprocessing Approach  
- Resized images to **256x256**, followed by center cropping to **224x224**.  
- Applied **data augmentation** (training only), including random rotation (10 degrees) and horizontal flipping for better generalization.  
- **Normalization** to scale pixel values with a mean and standard deviation of 0.5.  

### Model Architecture  
Five CNN-based architectures were experimented with for performance comparison:

1. **ResNet-18**
2. **ResNet-34**
4. **EfficientNet-B0**
5. **EfficientNet-B1**  
7. **Ensemble of the above four models**

All models were initialized with **pretrained ImageNet weights**, and the final fully connected layer was replaced to match the number of classes. Models were trained using **cross-entropy loss** and the **Adam optimizer**, with validation accuracy used for model selection. The ensemble employs **soft voting** by averaging the raw probability predictions from all four models for each input image to determine the final classification.

### Results 

#### AUC Scores  (in descending order of Macro AUC )
| Model            | No Substructure | Sphere Substructure | Vortex Substructure | Macro AUC | 
|-----------------|----------------|----------------------|----------------------|-------|
| **Ensemble**   | 0.9957         | 0.9935                | 0.9987                 | 0.9960 |
| **EfficientNet-B0**   | 0.9952           | 0.9909                 | 0.9971           | 0.9944 |
| **ResNet-34** | 0.9940      | 0.9906                 | 0.9976       | 0.9941    |
| **ResNet-18**   | 0.9936           | 0.9866                 | 0.9966                 | 0.9922 |
| **EfficientNet-B1**   | 0.9916           | 0.9857               | 0.9947      | 0.9907 |



### The model weights, inference notebooks and visualized results can be found inside this [directory](https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/tree/main/common_task). 

### Results and Analysis

Models were trained for 50 epochs with checkpoints saved at maximum validation accuracy. This strategy effectively mitigated overfitting and secured >95% validation accuracy for the ResNet and EfficientNet variants.

The following plots display the loss and accuracy trajectories for each model, illustrating their convergence rates and the effectiveness of the training strategy.



<table>
  <tr>
    <td align="center"><b> Models </b><br></td>
    <td align="center"><b> Training Curve </b><br></td>
    <td align="center"><b> Confusion Matrix </b><br></td>
  </tr>
  <tr>
    <td align="center"><b>EfficientNet-B1</b><br></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/efficientNet_b1_training.png" width="100%"></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/confusion_matrix/efficientNet_b1_cm.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>ResNet-34</b><br></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/resnet_34_training.png" width="100%"></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/confusion_matrix/resnet_34_cm.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>ResNet-18</b><br></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/resnet_18_training.png" width="100%"></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/confusion_matrix/resnet_18_cm.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>EfficientNet-B0</b><br></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/efficientNet_b0_training.png" width="100%"></td>
    <td align="center"><br><img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/confusion_matrix/efficientNet_b0_cm.png" width="100%"></td>
  </tr>
</table>


**Observation:** The training curves indicate that careful monitoring of both train and validation losses is crucial to prevent overfitting and to choose the optimal model state for deployment.

---
## Specific Test VII. Physics-Guided ML 

### My Approach 
The results are measured based on the following parameters:

1. Encoder:
  * I use a pre-trained ResNet-34 model as the primary encoder to extract features from the observed gravitational lensing images.
  * This encoder is specifically designed to predict the physical parameters—primarily the Einstein radius ($\theta_E$), necessary for the reconstruction phase.

2. Physics-Informed Lensing Layer:
  * I implement a custom differentiable Lensing Layer that incorporates the governing lens equation to compute angular deflections.
  * Using the predicted Einstein radius, the model analytically reconstructs the original source images by mapping pixels from the observed plane back to the source plane.
  * This step integrates domain-specific physical constraints into the deep learning pipeline, allowing the model to explicitly "undo" the distortion caused by gravitational lensing.

3. Classifier:
  * After reconstructing the source images, I use a pre-trained EfficientNet-b0 model solely on these reconstructed images.
  * The reconstructed images are passed through a dedicated encoder that performs the final classification task.
  * By isolating the source morphology from the lensing effects, the model can more accurately categorize the underlying astronomical objects based on their true physical structure.

![image](https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/specific_task/reconstructed_images.png)

### Results 

#### AUC Scores  (in descending order of Macro AUC )
| Model            | No Substructure | Sphere Substructure | Vortex Substructure | Macro AUC | 
|-----------------|----------------|----------------------|----------------------|-------|
| **Approach_1**   | 0.9990         | 0.9980                | 0.9998                 | 0.9989 |

<table>
  <tr>
    <td align="center"><b> ROC_AUC </b><br></td>
    <td align="center"><b> Training Curve </b><br></td>
    <td align="center"><b> Confusion Matrix </b><br></td>
  </tr>
  <tr>
    <td align="center"><br><img src=""></td>
    <td align="center"><br><img src=""></td>
    <td align="center"><br><img src="" width="100%"></td>
  </tr>
</table>
