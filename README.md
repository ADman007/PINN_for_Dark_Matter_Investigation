# PINN_for_Dark_Matter_Investigation
### DeepLense GSoC Assignment 
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
Two CNN-based architectures were experimented with for performance comparison:

1. **ResNet-18**
2. **ResNet-34**
3. **ResNet-50**
4. **EfficientNet-B0**
5. **EfficientNet-B1**  
6. **EfficientNet-B3**
7. **Ensemble of best ResNet and EfficientNet**

All models were initialized with **pretrained ImageNet weights**, and the final fully connected layer was replaced to match the number of classes. Models were trained using **cross-entropy loss** and the **Adam optimizer**, with validation accuracy used for model selection.  

### Results 

#### AUC Scores  (in descending order Mean )
| Model            | No Substructure | Sphere Substructure | Vortex Substructure | Mean | Val_Accuracy |
|-----------------|----------------|----------------------|----------------------|-------|-------|
| **ResNet-34**   | 0.99           | 0.99                 | 1.00                 | 0.00 |        |
| **EfficientNet-B3** | 0.99      | 0.99                 | 1.00                | 0.00 |          |
| **ResNet-34**   | 0.99           | 0.99                 | 1.00                 | 0.00 |        |
| **ResNet-34**   | 0.99           | 0.99                 | 1.00                 | 0.00 |        |
| **EfficientNet-B3**   | 0.99           | 0.99                 | 1.00                 | 0.00 |      |
| **ResNet-34**   | 0.99           | 0.99                 | 1.00                 | 0.00 |        |
| **Ensemble**   | 0.99           | 0.99                 | 1.00                 | 0.00 |         |
---  
