import { FMnistData } from "./fashion-data.js";
var canvas, ctx, saveButton, clearButton;
var pos = { x: 0, y: 0 };
var rawImage;
var model;

function getModel() {
  // In the space below create a convolutional neural network that can classify the
  // images of articles of clothing in the Fashion MNIST dataset. Your convolutional
  // neural network should only use the following layers: conv2d, maxPooling2d,
  // flatten, and dense. Since the Fashion MNIST has 10 classes, your output layer
  // should have 10 units and a softmax activation function. You are free to use as
  // many layers, filters, and neurons as you like.
  // HINT: Take a look at the MNIST example.
  model = tf.sequential();

  // YOUR CODE HERE
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 32,
      activation: "relu",
      kernel_initializer: "he_uniform",
    })
  );
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 32,
      activation: "relu",
      kernel_initializer: "he_uniform",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  // Compile the model using the categoricalCrossentropy loss,
  // the tf.train.adam() optimizer, and accuracy for your metrics.
  model.compile({
    // optimizer: tf.train.momentum(0.01, 0.9),
    optimizer: tf.train.adam(0.001), // Use Adam optimizer with a smaller learning rate

    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// ...

async function train(model, data) {
  // Set the following metrics for the callback: 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
  const metrics = ["loss", "val_loss", "acc", "val_acc"]; // YOUR CODE HERE

  // Create the container for the callback. Set the name to 'Model Training' and
  // use a height of 1000px for the styles.
  const container = { name: "Model Training", styles: { height: "1000px" } }; // YOUR CODE HERE

  // Use tfvis.show.fitCallbacks() to set up the callbacks.
  // Use the container and metrics defined above as the parameters.
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics); // YOUR CODE HERE

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 50000; // Increase training data size
  const TEST_DATA_SIZE = 10000; // Increase testing data size

  // Get the training batches and resize them. Remember to put your code
  // inside a tf.tidy() clause to clean up all the intermediate tensors.
  // HINT: Take a look at the MNIST example.
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  }); // YOUR CODE HERE

  // Get the testing batches and resize them. Remember to put your code
  // inside a tf.tidy() clause to clean up all the intermediate tensors.
  // HINT: Take a look at the MNIST example.
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  }); // YOUR CODE HERE
  let results = await model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 20, // Increase the number of epochs
    shuffle: true,
    callbacks: fitCallbacks,
  });
  // After training the model
  await model.save(
    tf.io.withSaveHandler(async (modelArtifacts) => {
      const modelJson = JSON.stringify(modelArtifacts, null, 2);

      // Save the model JSON as a file
      const blob = new Blob([modelJson], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "model2.json";
      a.click();
      URL.revokeObjectURL(url);
    })
  );
return model.fit(trainXs, trainYs, {
  batchSize: BATCH_SIZE,
  validationData: [testXs, testYs],
  epochs: 20, // Increase the number of epochs
  shuffle: true,
  callbacks: fitCallbacks,
});
}

// ...

// async function train(model, data) {
//   // Set the following metrics for the callback: 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
//   const metrics = ["loss", "val_loss", "acc", "val_acc"]; // YOUR CODE HERE

//   // Create the container for the callback. Set the name to 'Model Training' and
//   // use a height of 1000px for the styles.
//   const container = { name: "Model Training", styles: { height: "1000px" } }; // YOUR CODE HERE

//   // Use tfvis.show.fitCallbacks() to setup the callbacks.
//   // Use the container and metrics defined above as the parameters.
//   const fitCallbacks = tfvis.show.fitCallbacks(container, metrics); // YOUR CODE HERE

//   const BATCH_SIZE = 512;
//   const TRAIN_DATA_SIZE = 6000;
//   const TEST_DATA_SIZE = 1000;

//   // Get the training batches and resize them. Remember to put your code
//   // inside a tf.tidy() clause to clean up all the intermediate tensors.
//   // HINT: Take a look at the MNIST example.
//   const [trainXs, trainYs] = tf.tidy(() => {
//     const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
//     return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
//   }); // YOUR CODE HERE

//   // Get the testing batches and resize them. Remember to put your code
//   // inside a tf.tidy() clause to clean up all the intermediate tensors.
//   // HINT: Take a look at the MNIST example.
//   const [testXs, testYs] = tf.tidy(() => {
//     const d = data.nextTestBatch(TEST_DATA_SIZE);
//     return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
//   }); // YOUR CODE HERE

//   return model.fit(trainXs, trainYs, {
//     batchSize: BATCH_SIZE,
//     validationData: [testXs, testYs],
//     epochs: 10,
//     shuffle: true,
//     callbacks: fitCallbacks,
//   });
// }

function drawImage(image) {
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
  rawImage.src = canvas.toDataURL("image/png");
}

function handleImageUpload(e) {
  const file = e.target.files[0];
  const reader = new FileReader();

  reader.onload = function (event) {
    const img = new Image();
    img.onload = function () {
      // Draw the uploaded image on the canvas
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Update the rawImage src
      rawImage.src = canvas.toDataURL("image/png");
    };
    img.src = event.target.result;
  };

  reader.readAsDataURL(file);
}

function erase() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function save() {
  var raw = tf.browser.fromPixels(rawImage, 1);
  var resized = tf.image.resizeBilinear(raw, [28, 28]);
  var tensor = resized.expandDims(0);

  var prediction = model.predict(tensor);
  var pIndex = tf.argMax(prediction, 1).dataSync();

  var classNames = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
  ];

  alert(classNames[pIndex]);
}

function init() {
  canvas = document.getElementById("canvas");
  rawImage = document.getElementById("canvasimg");
  ctx = canvas.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);

  var uploadInput = document.getElementById("upload"); // Define uploadInput here
  uploadInput.addEventListener("change", handleImageUpload);

  saveButton = document.getElementById("sb");
  saveButton.addEventListener("click", save);
  clearButton = document.getElementById("cb");
  clearButton.addEventListener("click", erase);
}

async function run() {
  const data = new FMnistData();
  await data.load();
  model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture" }, model);
  await train(model, data);
  init();
  alert("Training is done, try classifying your uploaded images!");
}

document.addEventListener("DOMContentLoaded", run);
// function setPosition(e){
//     pos.x = e.clientX-100;
//     pos.y = e.clientY-100;
// }

// function draw(e) {
//     if(e.buttons!=1) return;
//     ctx.beginPath();
//     ctx.lineWidth = 24;
//     ctx.lineCap = 'round';
//     ctx.strokeStyle = 'white';
//     ctx.moveTo(pos.x, pos.y);
//     setPosition(e);
//     ctx.lineTo(pos.x, pos.y);
//     ctx.stroke();
//     rawImage.src = canvas.toDataURL('image/png');
// }

// function erase() {
//     ctx.fillStyle = "black";
//     ctx.fillRect(0,0,280,280);
// }

// function save() {
//     var raw = tf.browser.fromPixels(rawImage,1);
//     var resized = tf.image.resizeBilinear(raw, [28,28]);
//     var tensor = resized.expandDims(0);

//     var prediction = model.predict(tensor);
//     var pIndex = tf.argMax(prediction, 1).dataSync();

//     var classNames = ["T-shirt/top", "Trouser", "Pullover",
//                       "Dress", "Coat", "Sandal", "Shirt",
//                       "Sneaker",  "Bag", "Ankle boot"];

//     alert(classNames[pIndex]);
// }

// function init() {
//     canvas = document.getElementById('canvas');
//     rawImage = document.getElementById('canvasimg');
//     ctx = canvas.getContext("2d");
//     ctx.fillStyle = "black";
//     ctx.fillRect(0,0,280,280);
//     canvas.addEventListener("mousemove", draw);
//     canvas.addEventListener("mousedown", setPosition);
//     canvas.addEventListener("mouseenter", setPosition);
//     saveButton = document.getElementById('sb');
//     saveButton.addEventListener("click", save);
//     clearButton = document.getElementById('cb');
//     clearButton.addEventListener("click", erase);
// }

// async function run() {
//     const data = new FMnistData();
//     await data.load();
//     const model = getModel();
//     tfvis.show.modelSummary({name: 'Model Architecture'}, model);
//     await train(model, data);
//     // await model.save('downloads://my_model');
//     init();
//     alert("Training is done, try classifying your drawings!");
// }

// document.addEventListener('DOMContentLoaded', run);
