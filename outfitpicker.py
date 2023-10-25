# if not yet installed: 
#  pip install tkinter Pillow numpy matplotlib keras tensorflow requests
# if tensorflow module not found go to:
#  View -> command palette -> python: select interpreter -> chose the recommended python
# to run just simply click on the run/start button

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import tensorflow as tf
import requests
from datetime import datetime
from weatherSchema import weatherSchema

# Initialize Q-table
num_states = 3
num_actions = 27
Q = np.zeros((num_states, num_actions))

# Load the model
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Load the Fashion MNIST dataset
(_, _), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

outfits = {
    'casual': {
        'Sunny': {
            'top': ['T-shirt/top', 'Pullover'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker', 'Sandal']
        },
        'Rainy': {
            'top': ['Coat'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker']
        },
        'Stormy': {
            'top': ['Coat', 'Pullover'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker']
        },
        'Cloudy': {
            'top': ['Coat', 'T-shirt/top'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker', 'Sandal']
        }
    },
    'formal': {
        'Sunny': {
            'top': ['Shirt', 'Dress'],
            'bottom': ['Trouser'],
            'shoes': ['Ankle boot', 'Sandal']
        },
        'Rainy': {
            'top': ['Coat', 'Shirt', 'Dress'],
            'bottom': ['Trouser'],
            'shoes': ['Ankle boot']
        },
        'Stormy': {
            'top': ['Coat'],
            'bottom': ['Trouser'],
            'shoes': ['Ankle boot']
        },
        'Cloudy': {
            'top': ['Shirt', 'Dress'],
            'bottom': ['Trouser'],
            'shoes': ['Ankle boot', 'Sandal']
        }
    },
    'sporty': {
        'Sunny': {
            'top': ['T-shirt/top', 'Pullover'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker']
        },
        'Rainy': {
            'top': ['Coat', 'T-shirt/top'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker']
        },
        'Stormy': {
            'top': ['Coat'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker']
        },
        'Cloudy': {
            'top': ['T-shirt/top', 'Pullover'],
            'bottom': ['Trouser'],
            'shoes': ['Sneaker']
        }
    }
}

# Set hyperparameters
alpha = 0.1  # learning rate
gamma = 0.6  # discount factor
epsilon = 0.2  # exploration rate

# Simulated outfit recommendation environment
def get_current_state():
    return np.random.randint(0, num_states)

# Initialize global variables
state = get_current_state()
current_weather_code = 0
weatherCondition = ""


#get current weather condition
def fetch_current_weather_code():
    global current_weather_code
    global weatherCondition

    url = "https://api.open-meteo.com/v1/forecast?latitude=-36.8485&longitude=174.7635&hourly=weathercode&forecast_days=1"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'hourly' in data and 'weathercode' in data['hourly']:
            weather_codes = data['hourly']['weathercode']
            current_time = datetime.now()
            current_hour = current_time.hour
            if current_hour < len(weather_codes):
                current_weather_code = weather_codes[current_hour]
                # Find the dictionary in weatherSchema with matching code
                for weather_data in weatherSchema:
                    if weather_data["code"] == current_weather_code:
                        weatherCondition= weather_data['description']
                        weather_label.config(text=f"Current Hourly Weather Condition: {weather_data['description']}", font=('Arial', 20, 'bold'), fg='black')
                        break
                else:
                    weather_label.config(text="Weather code not found in the schema.")
            else:
                print("Current hour is not within the available range.")
        else:
            print("No weather data found in the response.")
    else:
        print(f"Request failed with status code {response.status_code}")



def recommend_outfit(outfits, occasion, current_weather):
    """
    Recommends an outfit based on the current weather condition and the specified occasion.
    Returns a ist of recommended outfit items based on the specified occasion and current weather condition.
    
    Parameters:
        outfits (dictionary): Dictionary containing outfit options for various occasions and weather conditions.
        occasion (str): Occasion for which the outfit is being recommended (e.g., 'casual', 'formal', 'sporty').
        current_weather (str): Current weather condition for which the outfit is being recommended.
    """

    if occasion in outfits and outfits[occasion] and current_weather in outfits[occasion]:
        outfit_options = outfits[occasion][current_weather]
    else:
        raise ValueError(f"No {occasion} outfit found for the current weather condition.")

    recommended_outfit = []

    for item in outfit_options:
        item_options = outfit_options[item]
        if item_options:
            item_index = np.random.randint(len(item_options))
            recommended_outfit.append(item_options[item_index])
        else:
            return {"error": f"No {item} outfit found for the current occasion."}

    return recommended_outfit


# Create the UI
root = tk.Tk()
root.title("Outfit Recommendation System")
root.geometry("600x500")
root.configure(bg='pink')
weather_panel = tk.Frame(root, bg='pink')
weather_panel.pack(side="top", fill="both", expand=True)
outfit_panel = tk.Frame(root, bg='pink')
outfit_panel.pack(side="left", fill="both", expand=True)

# Display the outfit in the UI
def display_outfit_ui(outfit_items, class_names, test_images, test_labels, panel, occasion):
    for widget in panel.winfo_children():
        widget.destroy()

    if occasion == "formal" and "Dress" in outfit_items:
        label_names = ["Dress", "Shoes"]
        outfit_items = [outfit_items[0], outfit_items[2]]
    else:
        label_names = ["Top", "Bottom", "Shoes"]

    # Add labels on the left
    for i, name in enumerate(label_names):
        tk.Label(panel, text=name, bg='pink', fg='black', font=('Arial', 12, 'bold')).grid(row=i, column=0, padx=5, pady=5)

    # Display outfit images
    for i, item in enumerate(outfit_items):
        if item:
            outfit_index = np.where(np.array(class_names) == item)[0][0]
            img = Image.fromarray(test_images[np.random.choice(np.where(test_labels == outfit_index)[0])])
            img = img.resize((100, 100))
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(panel, image=photo, bg='pink')
            label.image = photo
            label.grid(row=i, column=1, padx=5, pady=5)
            label.config(text=item, compound='top', bg='pink')


# Functions for handling like and dislike
def like_outfit():
    handle_feedback(1)

def dislike_outfit():
    handle_feedback(-1)

def handle_feedback(feedback):
    """
    Handles the feedback provided for the recommended outfit and updates the Q-table accordingly.
    The function also triggers the display of a new outfit recommendation based on the updated state.

    Parameter:
        feedback (int):  A positive value indicates a like, while a negative value indicates a dislike.
    """
    global state  # Use the global state variable

    # Update Q-table based on the feedback
    old_value = Q[state, 0]
    next_max = np.max(Q[state])
    new_value = (1 - alpha) * old_value + alpha * (feedback * 10 + gamma * next_max)
    Q[state, 0] = new_value

    # Update the state for the next recommendation
    state = get_current_state()

    chosen_occasion = selected_occasion.get()
    recommended_outfit = recommend_outfit(outfits,chosen_occasion, weatherCondition)
    if 'error' in recommended_outfit:
        print("error in handle_feedback " +recommended_outfit['error'])
    else:
        display_outfit_ui(recommended_outfit, class_names, test_images, test_labels, outfit_panel, chosen_occasion)


# Add like and dislike buttons and occasion dropdown on the UI
occasions = ['casual', 'formal', 'sporty']  # List of occasions
selected_occasion = tk.StringVar(root)
selected_occasion.set(occasions[0])  # Set the default occasion

control_panel = tk.Frame(root, bg='pink')
control_panel.pack(side="right", fill="both", expand=True)

# Buttons for submitting feedback
like_button = tk.Button(control_panel, text="Like", command=like_outfit, relief="raised")
like_button.config(bg='pink')
like_button.pack(side="top", padx=10, pady=10)

dislike_button = tk.Button(control_panel, text="Dislike", command=dislike_outfit, relief="raised")
dislike_button.config(bg='pink')
dislike_button.pack(side="top", padx=10, pady=10)

occasion_dropdown = tk.OptionMenu(control_panel, selected_occasion, *occasions)
occasion_dropdown.config(bg='pink', relief="flat")
occasion_dropdown.pack(side="top", padx=10, pady=10)


def update_occasion(*args):
    """
    Updates the selected occasion and triggers the recommendation of a new outfit based on the chosen occasion
    and the current weather condition.
    """
    chosen_occasion = selected_occasion.get()
    current_weather = weatherCondition
    try:
        recommended_outfit = recommend_outfit(outfits, chosen_occasion, current_weather)
        display_outfit_ui(recommended_outfit, class_names, test_images, test_labels, outfit_panel, chosen_occasion)
    except ValueError as e:
        print(e)

selected_occasion.trace("w", update_occasion)

weather_label = tk.Label(weather_panel, text="Weather code: ", bg='pink')
weather_label.pack(side="top", padx=10, pady=10)


if __name__ == '__main__':
    # Call functions to display initial outfit images
    chosen_occasion = selected_occasion.get()
    fetch_current_weather_code()
    current_weather = weatherCondition
    try:
        recommended_outfit = recommend_outfit(outfits, chosen_occasion, current_weather)
        display_outfit_ui(recommended_outfit, class_names, test_images, test_labels, outfit_panel, chosen_occasion)
    except ValueError as e:
        print(e)

    root.mainloop()