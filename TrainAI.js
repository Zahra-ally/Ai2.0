import { FMnistData } from "./TrainAI-fashion-data.js";
let image;
let model;

function createModel() {
  model = tf.sequential();

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

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function trainModel(model, data) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = { name: "Model Training", styles: { height: "1000px" } };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const batchSize = 512;
  const trainDataSize = 50000;
  const testDataSize = 10000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const batch = data.nextTrainBatch(trainDataSize);
    return [batch.xs.reshape([trainDataSize, 28, 28, 1]), batch.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const batch = data.nextTestBatch(testDataSize);
    return [batch.xs.reshape([testDataSize, 28, 28, 1]), batch.labels];
  });

  const results = await model.fit(trainXs, trainYs, {
    batchSize,
    validationData: [testXs, testYs],
    epochs: 20,
    shuffle: true,
    callbacks: fitCallbacks,
  });

  await model.save(
    tf.io.withSaveHandler(async (modelArtifacts) => {
      const modelJson = JSON.stringify(modelArtifacts, null, 2);

      const blob = new Blob([modelJson], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "model2.json";
      a.click();
      URL.revokeObjectURL(url);
    })
  );

  return results;
}


function initialize() {
  // Initialization code here
}

async function runModel() {
  const data = new FMnistData();
  await data.load();
  model = createModel();
  tfvis.show.modelSummary({ name: "Model Architecture" }, model);
  await trainModel(model, data);
  initialize();
  alert("Training is done, try classifying your uploaded images!");
}

document.addEventListener("DOMContentLoaded", runModel);
