const LOOKUP = [
  "T-shirt",
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

async function loadModel() {
  const model = await tf.loadLayersModel("my_model/model5.json");
  return model;
}

function getRandomImage(images) {
  return images[Math.floor(Math.random() * images.length)];
}

document
  .getElementById("recommendButton")
  .addEventListener("click", recommendOutfit);

async function recommendOutfit() {
  const model = await loadModel();
  const occasionSelect = document.getElementById("occasionSelect");
  const selectedOccasion = occasionSelect.value;

  const imagePaths = {
    casual: [
      //   "casual_image1.png",
      "casual_image2.png",
      "casual_image3.png",
      "casual_image4.png",
      "casual_image5.png",
      "casual_image6.png",
      "casual_image7.png",
    ],
    formal: [
      "formal_image1.png",
      "formal_image2.png",
      "formal_image3.png",
      "formal_image4.png",
      "formal_image5.png",
    ],
  };

  const outfitRecommendation = document.getElementById("outfitRecommendation");
  outfitRecommendation.innerHTML = "";

  const recommendations = { top: false, bottom: false, shoe: false };

  const loadImage = async (imagePath) => {
    return new Promise((resolve, reject) => {
      const imageElement = new Image();
      imageElement.src = imagePath;

      imageElement.onload = () => {
        resolve(imageElement);
      };

      imageElement.onerror = () => {
        reject("Image load error");
      };
    });
  };

  const images = await Promise.all(imagePaths[selectedOccasion].map(loadImage));

  images.forEach(async (imageElement) => {
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageElement, 0, 0, 28, 28);

    // Preprocess the image as needed.
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const inputTensor = tf.browser.fromPixels(imageData, 1);
    const normalizedInput = inputTensor.toFloat().div(255.0); // Normalize and convert to float.
    const flattenedInput = normalizedInput.reshape([1, 784]); // Flatten the image.

    // Perform inference using the loaded model.
    const prediction = model.predict(flattenedInput);

    // Get the predicted class index.
    const predictedClass = prediction.argMax(1).dataSync()[0];

    // Map the predicted class index to the class label.
    const predictedLabel = LOOKUP[predictedClass];

    if (
      (predictedLabel == "T-shirt" ||
        predictedLabel == "Pullover" ||
        predictedLabel == "Coat" ||
        predictedLabel == "Shirt") &&
      !recommendations.top
    ) {
      // Display the image and the predicted label.
      const imageContainer = document.createElement("div");
      const imgElement = new Image();
      imgElement.src = imageElement.src;
      const labelElement = document.createElement("p");
      labelElement.innerText = `Predicted Label: ${predictedLabel}`;

      imageContainer.appendChild(imgElement);
      imageContainer.appendChild(labelElement);

      outfitRecommendation.appendChild(imageContainer);
      recommendations.top = true;
    }

    if (predictedLabel == "Trouser" && !recommendations.bottom) {
      // Display the image and the predicted label.
      const imageContainer = document.createElement("div");
      const imgElement = new Image();
      imgElement.src = imageElement.src;
      const labelElement = document.createElement("p");
      labelElement.innerText = `Predicted Label: ${predictedLabel}`;

      imageContainer.appendChild(imgElement);
      imageContainer.appendChild(labelElement);

      outfitRecommendation.appendChild(imageContainer);
      recommendations.bottom = true;
    }

    if (
      predictedLabel == "Dress" &&
      !recommendations.bottom &&
      !recommendations.top
    ) {
      // Display the image and the predicted label.
      const imageContainer = document.createElement("div");
      const imgElement = new Image();
      imgElement.src = imageElement.src;
      const labelElement = document.createElement("p");
      labelElement.innerText = `Predicted Label: ${predictedLabel}`;

      imageContainer.appendChild(imgElement);
      imageContainer.appendChild(labelElement);

      outfitRecommendation.appendChild(imageContainer);
      recommendations.bottom = true;
      recommendations.top = true;
    }

    if (
      (predictedLabel == "Sandal" ||
        predictedLabel == "Sneaker" ||
        predictedLabel == "Ankle boot") &&
      !recommendations.shoe
    ) {
      // Display the image and the predicted label.
      const imageContainer = document.createElement("div");
      const imgElement = new Image();
      imgElement.src = imageElement.src;
      const labelElement = document.createElement("p");
      labelElement.innerText = `Predicted Label: ${predictedLabel}`;

      imageContainer.appendChild(imgElement);
      imageContainer.appendChild(labelElement);

      outfitRecommendation.appendChild(imageContainer);
      recommendations.shoe = true;
    }
  });
}
