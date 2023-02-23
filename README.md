# Image Classification

This is a simple FastAPI application that provides an endpoint to classify images based on a pre-trained model.



## How to Train the Model

To train the model, run the `model.py` script:

```
python model.py
```

The model is saved under the file `model\model.h5`.

## How to Run the Application

Put the image you want to predict in the `image-classification` folder (example: `image.jpg`).


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





