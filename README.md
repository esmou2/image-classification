# Image Classification

This is a simple FastAPI application that provides an endpoint to classify images based on a pre-trained model.


## Requirements
* Python 3.9 or higher
* Docker
* FastAPI
* TensorFlow
* NumPy
* scikit-learn
You can install the required Python packages by running the following command:

```
pip install -r requirements.txt
```
Make sure to run this command from the root directory of the project, where the requirements.txt file is located.

## How to Train the Model
A copy of the model is already in the root directory but if you want to re train the model follow those steps:
1. To train the model, unzip the dataset in the repository folder(`Dataset\Data for test)
2. Run the `model.py` script:

```
python model/train_model.py
```

The model is saved under the file `model\model.h5`.

## How to Run the Application

Put the image you want to predict in the root directory (example: `image.jpg`).


### Build the Docker Image

To build the Docker image, navigate to the project directory and run the following command:

```
docker build -t api-image-classification .
```


To run the application, execute the following command:

```
docker run -p 5000:5000 api-image-classification
```

## API Endpoint

The application provides an API endpoint that can be used to classify images as `Bed, Chair or Sofa`. To use the endpoint, make a POST request to `/predict` with the following payload:
```
{
"image_path": "image.jpg"
}
```

```
The endpoint will return a JSON object with the prediction:
{
"Prediction": "This image most likely belongs to [class name] with a [confidence] percent confidence."
}
```





