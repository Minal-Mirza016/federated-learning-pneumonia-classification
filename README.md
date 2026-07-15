# Federated Learning for Pneumonia Detection Using TensorFlow

A deep learning project that implements Federated Learning for classifying chest X-ray images as **Pneumonia** or **Normal** using TensorFlow and Keras. The project simulates collaborative learning across multiple hospitals while preserving data privacy.

---

## Project Overview

Traditional machine learning models require collecting all data into a central location, which raises privacy concerns in healthcare.

This project demonstrates how **Federated Learning** enables multiple hospitals to collaboratively train a shared deep learning model without exchanging patient data. Each client trains locally, and only model parameters are shared and aggregated using the **FedAvg (Federated Averaging)** algorithm.

The trained global model is evaluated on chest X-ray images for pneumonia detection.

---

## Key Features

- Federated Learning implementation
- Multi-client (hospital) simulation
- FedAvg aggregation algorithm
- Chest X-ray image classification
- Deep Learning using TensorFlow and Keras
- Global model evaluation
- Confusion Matrix and Classification Report
- Model saving and reloading
- Prediction on unseen X-ray images

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab

---

## Dataset

- Chest X-ray Pneumonia Dataset
- Binary Classification
  - Normal
  - Pneumonia

---

## Model Workflow

1. Load and preprocess chest X-ray images
2. Simulate multiple hospital clients
3. Train local models
4. Aggregate model weights using FedAvg
5. Update the global model
6. Evaluate the global model
7. Generate predictions on unseen images
8. Save and reload the trained model

---

## Results

The trained federated model achieved promising performance on chest X-ray classification.

Evaluation includes:

- Validation Accuracy
- Test Accuracy
- Confusion Matrix
- Classification Report
- Sample Predictions

---

## Project Structure

```
## Project Structure

```text
federated-learning-pneumonia-classification
│
├── federated_learning.ipynb    # Jupyter Notebook containing the complete implementation
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

---

## How to Run

1. Clone or download this repository.
2. Open `federated_learning.ipynb` in Google Colab or Jupyter Notebook.
3. Install the required Python libraries if prompted.
4. Run all notebook cells sequentially.
5. Evaluate the trained federated learning model.

---

## Learning Outcomes

Through this project, I gained practical experience in:

- Federated Learning
- Deep Learning
- Medical Image Classification
- TensorFlow and Keras
- Model Evaluation
- Healthcare AI
- Collaborative Machine Learning

---

## Future Improvements

- Increase the number of simulated clients
- Train on real distributed hospital datasets
- Improve model generalization
- Deploy the trained model using Flask or FastAPI
- Explore Secure Aggregation techniques

---

## Author

**Minal Mirza**

Computer Science Student

Interested in Artificial Intelligence, Machine Learning, Data Science, Computer Vision, and Medical AI.

---

## License

This project is licensed under the MIT License.
