let weatherConditionAPI;
let weatherID;
let currentWeather = "Sunny"; // Set the default weather condition

fetch(
  `https://api.openweathermap.org/data/2.5/weather?q=Auckland&units=metric&appid=108dd9a67c96f23039937fe6f3c91963`
)
  .then((response) => response.json())
  .then((data) => {
    // Handle the weather data here (e.g., display temperature, description, etc.).
    console.log(data.weather[0].id);
    weatherID = data.weather[0].id;
    weatherConditionAPI = data.weather[0].main;

    labelElement.innerText = `Current Weather in Auckland: ${weatherConditionAPI}`;

    if (weatherID == 800 || weatherID == 801) {
      currentWeather = "Sunny";
    }
    if (weatherID == 802 || weatherID == 803 || weatherID == 804) {
      currentWeather = "Cloudy";
    }

    if (
      weatherID == 500 ||
      weatherID == 501 ||
      weatherID == 502 ||
      weatherID == 503 ||
      weatherID == 504
    ) {
      currentWeather = "Rainy";
    }
  })
  .catch((error) => {
    console.error("Error fetching weather data:", error);
  });

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

const QTable = {}; // Initialize the Q-table
const learningRate = 0.1;
const discountFactor = 0.9;

LOOKUP.forEach((outfit) => {
  QTable[outfit] = 0; // Initialize Q-values for each outfit
});

async function loadModel() {
  const model = await tf.loadLayersModel("my_model/model6.json");
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
      "casual_image2.png",
      "casual_image3.png",
      "casual_image4.png",
      "casual_image5.png",
      "casual_image6.png",
      "casual_image7.png",
      "casual_image8.png",
      "casual_image9.png",
      "casual_image10.png",
      "casual_image11.png",
      "casual_image12.png",
      "casual_image13.png",
    ],
    formal: [
      "formal_image1.png",
      "formal_image2.png",
      "formal_image3.png",
      "formal_image4.png",
      "formal_image5.png",
      "formal_image6.png",
      "formal_image7.png",
      "formal_image8.png",
      "formal_image9.png",
      "formal_image10.png",
    ],
    sportswear: [
      "sports_image1.png",
      "sports_image2.png",
      "sports_image3.png",
      "sports_image4.png",
      "sports_image5.png",
      "sports_image6.png",
      "sports_image7.png",
      "sports_image8.png",
      "sports_image9.png",
      "sports_image10.png",
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

    const imageData = ctx.getImageData(0, 0, 28, 28);
    const inputTensor = tf.browser.fromPixels(imageData, 1);
    const normalizedInput = inputTensor.toFloat().div(255.0);
    const flattenedInput = normalizedInput.reshape([1, 28, 28, 1]);

    const prediction = model.predict(flattenedInput);
    const predictedClass = prediction.argMax(1).dataSync()[0];
    const predictedLabel = LOOKUP[predictedClass];
    qLearning(
      predictedClass,
      currentWeather,
      selectedOccasion,
      recommendations
    );
    if (
      !recommendations.shoe &&
      (predictedLabel === "Sandal" ||
        predictedLabel === "Sneaker" ||
        predictedLabel === "Ankle boot")
    ) {
      displayOutfit(
        imageElement,
        predictedLabel,
        outfitRecommendation,
        recommendations,
        "shoe"
      );
    } else if (
      !recommendations.dress &&
      predictedLabel === "Dress" &&
      !recommendations.bottom &&
      !recommendations.top
    ) {
      displayOutfit(
        imageElement,
        predictedLabel,
        outfitRecommendation,
        recommendations,
        "dress"
      );
    } else if (
      !recommendations.top &&
      (predictedLabel === "T-shirt" ||
        predictedLabel === "Pullover" ||
        predictedLabel === "Coat" ||
        predictedLabel === "Shirt")
    ) {
      displayOutfit(
        imageElement,
        predictedLabel,
        outfitRecommendation,
        recommendations,
        "top"
      );
    } else if (
      !recommendations.bottom &&
      predictedLabel === "Trouser" &&
      !recommendations.dress &&
      !recommendations.top
    ) {
      displayOutfit(
        imageElement,
        predictedLabel,
        outfitRecommendation,
        recommendations,
        "bottom"
      );
    }
  });
}

function displayOutfit(
  imageElement,
  predictedLabel,
  outfitRecommendation,
  recommendations,
  outfitType
) {
  const imageContainer = document.createElement("div");
  const imgElement = new Image();
  imgElement.src = imageElement.src;
  const labelElement = document.createElement("p");

  imageContainer.appendChild(imgElement);
  imageContainer.appendChild(labelElement);

  outfitRecommendation.appendChild(imageContainer);
  recommendations[outfitType] = true;
}

function qLearning(
  predictedLabel,
  currentWeather,
  selectedOccasion,
  recommendations
) {
  // Set the exploration rate (epsilon) to control exploration vs. exploitation
  const epsilon = 0.2; 

  // Randomly explore with probability epsilon
  if (Math.random() < epsilon) {
    // Choose a random outfit for exploration
    const randomOutfitIndex = Math.floor(Math.random() * LOOKUP.length);
    return LOOKUP[randomOutfitIndex];
  } else {
    // Filter allowed outfits based on weather, occasion, and outfit type
    const allowedOutfits = LOOKUP.filter((outfit) => {
      if (
        (outfit === "T-shirt" || outfit === "Shirt") &&
        recommendations.top === false
      ) {
        return true;
      }
      if (outfit === "Trouser" && recommendations.bottom === false) {
        return true;
      }
      if (
        (outfit === "Sandal" ||
          outfit === "Sneaker" ||
          outfit === "Ankle boot") &&
        recommendations.shoe === false &&
        (currentWeather === "Sunny" ||
          currentWeather === "Cloudy" ||
          currentWeather === "Rainy")
      ) {
        return true;
      }
      if (
        outfit === "Dress" &&
        recommendations.dress === false &&
        (selectedOccasion === "casual" ||
          currentWeather === "Sunny" ||
          currentWeather === "Rainy")
      ) {
        return true;
      }
      return false;
    });

    // Choose the outfit with the highest Q-value among allowed outfits for exploitation
    const bestOutfit = allowedOutfits.reduce((best, outfit) => {
      return QTable[outfit] > QTable[best] ? outfit : best;
    }, allowedOutfits[0]);

    return bestOutfit;
  }
}
