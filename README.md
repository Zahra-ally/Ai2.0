# visualizing-fashion-mnist-on-browser
![img](https://miro.medium.com/max/1400/1*R6wGKAXL06jGL3tWRT_S-w.jpeg)
Training Fashion MNIST dataset on browser using tensorflowjs and visualizing model training using tfvisualizer js library.

# Inferencing Fashion MNIST Model from browser
* Draw something to predict, seems more fun than giving sample image.
![img](https://miro.medium.com/max/1400/1*L0-l2MDkf7qSccrcSvKguQ.jpeg)

* Visualize javascript training process directly on web page.
![img](https://miro.medium.com/max/1400/1*qNZE60y3XPVfP0kOum1rLA.jpeg)

# Running file:
Even though training occurs on static html files, tfjs libraries requires a server for serving the files. On website server is provied by hosting provider. 

Running file locally is also easy, as we need a server. There are many ways to serve http files like WAMP for windows or LAMP for linux, or chrome web http extension.

The simplest way to serve these file is by running one line python command inside folder containg files.

## Run python server inside project folder
* Python http module is built in so there is no need of installing additional package. Plus it is most simple way of serving static pages locally.

```bash
python3 -m http.server
```
 The above command will give following output:
 ```
 Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
 ```

 ## Open browser and open url: http://0.0.0.0:8000

## You will see html file along with other files on chrome browser

## Click on html file
This will open html file and start loading the tfjs.

# After opening file the training will start along with visualizer. Wait for training to finish. Once the training is finished an alert will pop up showing training is done.
# Now you can try drawing an input inside canvas after that click on classify to identify class from given input.