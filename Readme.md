# Iris Flower Predictor

This is a simple machine learning web application that predicts the species of an Iris flower based on its sepal and petal dimensions. The app is built using **Streamlit** and leverages the **Iris dataset** for training and evaluation.

## Features
- Input sepal and petal dimensions to predict the Iris flower species.
- Uses a **Decision Tree Classifier** for predictions.
- Visualizes the model's performance using a **confusion matrix**.
- Deployed using **Streamlit Community Cloud**.

## Dataset
The Iris dataset contains 150 samples of iris flowers with the following features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target labels:
- `Setosa`
- `Versicolor`
- `Virginica`

## Project Structure
```
|-- iris_app.py          # Main application script
|-- requirements.txt     # Dependencies
|-- README.md            # Project documentation
```

## Requirements
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/Abdul-Wasih03/iris-flower-predictor.git
   cd iris-flower-predictor
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run iris_app.py
   ```

3. Open the app in your browser at `http://localhost:8501`.

## Deployment
The app is deployed on **Streamlit Community Cloud**. Access it [here](https://abdul-wasih03-iris-flower-predictor.streamlit.app/).

## Model Training
- The model is a **Decision Tree Classifier** trained on the Iris dataset.
- Features are standardized using **StandardScaler** for better performance.

## Visualizations
- A confusion matrix is used to evaluate model accuracy.
- Built-in Streamlit widgets allow dynamic user input.

## How to Use the App
1. Open the app in your browser.
2. Enter the values for Sepal Length, Sepal Width, Petal Length, and Petal Width.
3. Click the **Predict** button.
4. The app will display the predicted species of the Iris flower.

## Screenshots
### Input Fields
![input](https://github.com/user-attachments/assets/80c9ac83-9ccb-4c46-9745-f8de2d5dda3f)

### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/9e0e4911-49d9-4be5-825c-f60d377cb9bd)
## Acknowledgments
- Dataset: [UCI Machine Learning Repository - Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- Built with [Streamlit](https://streamlit.io/)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
