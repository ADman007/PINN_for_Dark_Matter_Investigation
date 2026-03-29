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


### The model weight and inference notebooks can be found inside the directory. 

### Results and Analysis

During the training of both models, it was observed that the models started to overfit after approximately 10 epochs. The models were saved at the point where the gap between the train loss and the validation loss was minimal. This strategy helped achieve an accuracy of around 73% on the validation set for both models.

Below are the training curves for the VGG12 and LeNet architectures, illustrating the point of overfitting and the epoch at which the models were saved.

| Model Architecture | Training Curve |
|-------------------|----------------|
| VGG12             | <img src="https://github.com/Vishak-Bhat30/ML4SCI_24/assets/102585626/79c1cd93-921a-4ad4-ab92-656ba3d24f43" width="450" alt="Training Curve"> |
| LeNet             |<img src="https://github.com/Vishak-Bhat30/ML4SCI_24/assets/102585626/3c486ed0-c3d8-4e37-8ed7-87f5818cbfa3" width="450" alt="Training Curve"> |
| ResNet             |<img src="https://github.com/Vishak-Bhat30/ML4SCI_24/assets/102585626/12cbc9a7-317a-4949-ba66-4a28a31ee367" width="450" alt="Training Curve"> |
| DenseNet             |<img src="https://github.com/Vishak-Bhat30/ML4SCI_24/assets/102585626/9bd8f6ed-6029-47a4-8ddf-0298b67cd07a" width="450" alt="Training Curve"> |
| MobileNet             |<img src="https://github.com/Vishak-Bhat30/ML4SCI_24/assets/102585626/c929091c-d862-4084-a63a-13f4d658c1a8" width="450" alt="Training Curve"> |
| EfficientNet             |<img src="https://github.com/Vishak-Bhat30/ML4SCI_24/assets/102585626/f263e48a-6e07-4d10-ba93-46de7cbe77cc" width="450" alt="Training Curve"> |


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

