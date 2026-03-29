# <p align="center"> ML4SCI_26 </p>
### DeepLense - Physics Guided Machine Learning on Real Lensing Images
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
7. **Ensemble of best ResNet and EfficientNet**

All models were initialized with **pretrained ImageNet weights**, and the final fully connected layer was replaced to match the number of classes. Models were trained using **cross-entropy loss** and the **Adam optimizer**, with validation accuracy used for model selection.  

### Results 

#### AUC Scores  (in descending order of Macro AUC )
| Model            | No Substructure | Sphere Substructure | Vortex Substructure | Macro AUC | Test_Accuracy |
|-----------------|----------------|----------------------|----------------------|-------|-------|
| **EfficientNet-B1**   | 0.99           | 0.99                 | 1.00      | 0.9944 | 0.9636 |
| **ResNet-34** | 0.99      | 0.99                 | 1.00       | 0.9930    | 0.9544 |
| **ResNet-18**   | 0.99           | 0.99                 | 1.00                 | 0.9925 |  0.9507   |
| **EfficientNet-B0**   | 0.98           | 0.97                 | 0.99           | 0.9821 | 0.8909  |
| **Ensemble**   | 0.00          | 0.00                | 0.00                 | 0.00 |   0.00    |


### The model weights, inference notebooks and visualized results can be found inside this [directory](https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/tree/main/common_task). 

### Results and Analysis

During the training of both models, it was observed that the models started to overfit after approximately 10 epochs. The models were saved at the point where the gap between the train loss and the validation loss was minimal. This strategy helped achieve an accuracy of around 73% on the validation set for both models.

Below are the training curves for the VGG12 and LeNet architectures, illustrating the point of overfitting and the epoch at which the models were saved.

| Model Architecture | Training Curve |
|-------------------|----------------|
| **EfficientNet_b1**            | <img src="https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/efficientNet_b1_training.png" width="450" alt="Training Curve"> |
| **ResNet-34**             |<img src="[https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/resnet_34_training.png]" width="450" alt="Training Curve"> |
| **ResNet-18**             |<img src="[https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/resnet_18_training.png]" width="450" alt="Training Curve"> |
| **EfficientNet_b0**             |<img src="[https://github.com/ADman007/PINN_for_Dark_Matter_Investigation/blob/main/common_task/training_curves/efficientNet_b0_training.png]" width="450" alt="Training Curve"> |


**Observation:** The training curves indicate that careful monitoring of both train and validation losses is crucial to prevent overfitting and to choose the optimal model state for deployment.

---
## Specific Test VII. Physics-Guided ML 

### Findings  
The results are measured based on the following parameters:

- `dtype`: `float16`
- `torch.compile` with mode `reduce-overhead`.
- ODE Solver used is `Euler` with `50` timesteps.   

| Model           | FID  | ODE Timesteps | Latency | Batch Size |
|---------------|------|---------------|---------|------------|
| **Transformer** | 27.6961 | 50          | 109ms   | 1          |
| **UNet**      | 34.8437 | 50            | 123ms   | 1          |

### Model Architecture  
A Flow Matching model was implemented and trained using two different backbones for performance comparison:

1. **UNet Backbone**  
   - Utilizes a ResNet-34 encoder with pretrained ImageNet weights.  

2. **Transformer Backbone**  
   - Based on a Vision Transformer (ViT) architecture with pretrained ImageNet weights.  

