# Advanced Topics in Data Science, Machine Learning, and Deep Learning

Note: All work in this repository is authored by Darien Nouri.


This repo contains a collection of notebooks pertaining to various advanced topics in ML and deep learning that were created as part of NYU studies.


## File Structure

```
├── README.md
├── 00_Bias_Variance_Tradeoff.ipynb
├── 02_LogisticRegression_Regularization.ipynb
├── 03_Algorithmic_Performance_Scaling.ipynb
├── 04_Perceptron.ipynb
├── 05_Linear_Separability.ipynb
├── 06_Softmax_Activation_Derivation.ipynb
├── 07_NeuralNetwork_Manual_Backpropagation.ipynb
├── 08_Weight_Initialization_DeadNeurons_LeaklyReLU.ipynb
├── 09_BatchNorm_Dropout.ipynb
├── 10_LearningRate_BatchSIze_Exploration.ipynb
├── 11_Convolutional_NN_Architectures.ipynb
├── 12_TransferLearning_Shallow_vs_Finetuning.ipynb
├── 13_SiameseNetwork_Facial_Recognition.ipynb
├── 14_Sentiment_Analysis_RNNs.ipynb
├── 15_Seq_to_Seq_Chatbot_Training.ipynb
├── 16_Attention_in_Transformers.ipynb
├── 17_H2O_Hyperparameter_Optimization.ipynb
├── 18_Bert_LLM_Finetuning.ipynb
├── 19_Auto_Feature_Engineering.ipynb
├── 20_LeNet_RayTune_Hyperparameter_Optimization.ipynb
├── 21_PyTorch_DataParallelism.ipynb
├── 22_Staleness_ParameterServer_AsyncSGD.ipynb
└── 23_SSD_ONNX_Object_Detection.ipynb
```

## Notebooks
- **23_SSD_ONNX_Object_Detection.ipynb**
  - Explores inferencing using the SSD ONNX model with the ONNX Runtime Server.
  - Includes model testing, fine-tuning, conversion to ONNX, and running inferencing using ONNX Runtime.

- **22_Staleness_ParameterServer_AsyncSGD.ipynb**
  - Calculates staleness in a Parameter-Server based Asynchronous SGD training system with two learners.
  - Examines the number of weight updates between reading and updating weights for each gradient calculation.

- **21_PyTorch_DataParallelism.ipynb**
  - Experiments with PyTorch's DataParallel Module for Synchronous SGD across multiple GPUs.
  - Analyzes training time, scalability, and communication bandwidth utilization.

- **20_LeNet_RayTune_Hyperparameter_Optimization.ipynb**
  - Compares Grid Search, Bayesian Search, and Hyperband for hyperparameter optimization using Ray Tune on the MNIST dataset.
  - Measures time efficiency and model performance.

- **19_Auto_Feature_Engineering.ipynb**
  - Demonstrates the use of AutoFeat for automated feature engineering and selection on a regression dataset.
  - Includes interpretability discussions, feature selection, model training, and evaluation.

- **18_Bert_LLM_Finetuning.ipynb**
  - Fine-tunes BERT for a question-answering task.
  - Includes loading BERT, training, and evaluating its performance.

- **17_H2O_Hyperparameter_Optimization.ipynb**
  - Compares H2O's grid search, randomized grid search, and AutoML for hyperparameter optimization.
  - Evaluates model performance and identifies the best hyperparameters.

- **16_Attention_in_Transformers.ipynb**
  - Explains the self-attention mechanism in Transformers.
  - Covers the calculation of softmax scores, multi-headed attention, and combining multiple heads.

- **15_Seq_to_Seq_Chatbot_Training.ipynb**
  - Trains a simple chatbot using the Cornell Movie Dialogs Corpus and a sequence-to-sequence model with Luong attention.
  - Includes hyperparameter sweeps with Weights and Biases (W&B).

- **14_Sentiment_Analysis_RNNs.ipynb**
  - Compares the performance of RNN, LSTM, GRU, and BiLSTM for sentiment analysis using the IMDB dataset.
  - Analyzes the accuracy of each model.

- **13_SiameseNetwork_Facial_Recognition.ipynb**
  - Trains a Siamese network for face recognition and evaluates its performance with different contrastive loss functions.
  - Examines robustness with and without glasses and compares Mining-Contrastive loss.

- **12_TransferLearning_Shallow_vs_Finetuning.ipynb**
  - Explores transfer learning for image classification using a pre-trained ResNet50 model.
  - Compares fine-tuning the model and using it as a fixed feature extractor.

- **11_Convolutional_NN_Architectures.ipynb**
  - Studies and compares different convolutional neural network architectures.
  - Includes parameter calculations, memory requirements, inception modules analysis, and Faster R-CNN.

- **10_LearningRate_BatchSIze_Exploration.ipynb**
  - Explores the cyclical learning rate policy and its effect on training a neural network.
  - Compares the effect of varying batch sizes with a fixed learning rate on model performance.

- **09_BatchNorm_Dropout.ipynb**
  - Compares the effects of batch normalization and dropout on the performance of LeNet-5 using the MNIST dataset.
  - Investigates the combination of both techniques and their impact on performance.

- **08_Weight_Initialization_DeadNeurons_LeaklyReLU.ipynb**
  - Explores the effects of weight initialization, vanishing gradients, and dead neurons.
  - Analyzes the impact of ReLU and Leaky ReLU activations.

- **07_NeuralNetwork_Manual_Backpropagation.ipynb**
  - Manually implements a 3-layered neural network with scaled sigmoid activation functions.
  - Covers forward propagation, cost calculation, backpropagation, model training, and accuracy comparison.

- **06_Softmax_Activation_Derivation.ipynb**
  - Derives the properties of the softmax activation function and its derivatives.
  - Uses cross-entropy loss function and proves the gradient of the loss function.

- **05_Linear_Separability.ipynb**
  - Examines linear separability in a dataset with two features.
  - Includes dataset analysis, feature transformation, separating hyperplane, and importance of nonlinear transformations.

- **04_Perceptron.ipynb**
  - Implements the perceptron algorithm and explores different loss functions.
  - Includes dataset generation, training perceptron models, accuracy evaluation, and performance comparison.

- **03_Algorithmic_Performance_Scaling.ipynb**
  - Studies algorithmic performance scaling using a large classification dataset.
  - Involves dataset summary, model training, learning curves, training time analysis, and performance comparison.

- **02_LogisticRegression_Regularization.ipynb**
  - Investigates regularization techniques in logistic regression using the IRIS dataset.
  - Covers parameter significance, regularization penalties, model fitting, and coefficient analysis.

- **00_Bias_Variance_Tradeoff.ipynb**
  - Explores the bias-variance tradeoff in a regression problem.
  - Includes dataset generation, polynomial estimators, bias-variance tradeoff analysis, and model selection.