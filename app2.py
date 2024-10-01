import streamlit as st
import pymongo
import numpy as np
import tensorflow as tf
from passlib.hash import bcrypt
from PIL import Image
from fuzzywuzzy import process
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from fuzzywuzzy import fuzz
import cv2
import requests

# Load pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
model.trainable = False



# Function to connect to MongoDB and fetch image URLs based on keywords
def fetch_image_urls(keyword):
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb+srv://abhay21csu439:gangala12@recommendation.hoyhukz.mongodb.net/")
        db = client.get_database("recommendation-system")
        collection = db.get_collection("dataset")

        # Fetch image URLs based on keywords in description or articleType
        query = {"$or": [
            {"productDisplayName": {"$regex": keyword, "$options": "i"}},
            {"articleType": {"$regex": keyword, "$options": "i"}}
        ]}
        cursor = collection.find(query, {"image_url": 1})
        image_urls = [doc["image_url"] for doc in cursor]

        # Close the MongoDB client
        client.close()

        return image_urls
    except Exception as e:
        print(f"Error accessing MongoDB: {e}")
        return []

# Function to fetch additional details from MongoDB based on the selected image URL
def fetch_image_details(image_url):
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb+srv://abhay21csu439:gangala12@recommendation.hoyhukz.mongodb.net/")
        db = client.get_database("recommendation-system")
        collection = db.get_collection("dataset")

        # Fetch details based on the selected image URL
        details = collection.find_one({"image_url": image_url})

        # Close the MongoDB client
        client.close()

        return details
    except Exception as e:
        print(f"Error fetching image details: {e}")
        return None

# Function to compute image embeddings using a pre-trained ResNet50 model
def compute_image_embedding(image_url):
    try:
        # Fetch image from URL and preprocess it
        img = tf.keras.utils.get_file("temp_img.jpg", image_url)
        img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))

        # Compute image embedding
        embedding = model.predict(img_array)

        return embedding.flatten()
    except Exception as e:
        print(f"Error computing image embedding: {e}")
        return None

# Function to compute color histogram similarity
def compute_color_histogram_similarity(img1, img2):
    # Convert images to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Calculate histograms for both images
    hist1 = cv2.calcHist([img1_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    # Compute histogram similarity (using correlation)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return similarity

# Function to compute pattern similarity (using ORB feature detection)
def compute_pattern_similarity(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # Initialize matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Compute pattern similarity (using number of matches)
    similarity = len(matches)
    
    return similarity

# Function to recommend related products based on color and pattern similarity
def recommend_related_products(selected_embedding, all_embeddings, all_image_urls, top_n=5):
    try:
        # Initialize lists to store similarity scores
        color_similarities = []
        pattern_similarities = []
        total_similarities = []

        # Compute color and pattern similarity for each product
        for emb, url in zip(all_embeddings, all_image_urls):
            # Load images
            selected_img = cv2.imread(url)
            other_img = cv2.imread(url)
            
            # Compute color similarity
            color_similarity = compute_color_histogram_similarity(selected_img, other_img)
            color_similarities.append(color_similarity)
            
            # Compute pattern similarity
            pattern_similarity = compute_pattern_similarity(selected_img, other_img)
            pattern_similarities.append(pattern_similarity)
            
            # Combine color and pattern similarities (you can adjust weights as needed)
            total_similarity = 0.7 * color_similarity + 0.3 * pattern_similarity
            total_similarities.append(total_similarity)

        # Sort products based on total similarity
        sorted_indices = np.argsort(total_similarities)[::-1]

        # Select top N most similar products based on color
        related_products = [(all_image_urls[idx], total_similarities[idx]) for idx in sorted_indices[:top_n]]

        return related_products
    except Exception as e:
        print(f"Error recommending related products: {e}")
        return []

# Function to refine recommendations based on design similarity
def refine_recommendations(selected_embedding, all_embeddings, all_image_urls, top_n=5):
    try:
        # Compute design similarity matrix between selected embedding and all other embeddings
        design_similarity_matrix = pairwise_distances(all_embeddings, selected_embedding.reshape(1, -1), metric='cosine')

        # Sort products based on design similarity
        sorted_indices = np.argsort(design_similarity_matrix.flatten())

        # Select top N most similar products
        refined_recommendations = [(all_image_urls[idx], 1 - design_similarity_matrix[idx][0]) for idx in sorted_indices[:top_n]]

        return refined_recommendations
    except Exception as e:
        print(f"Error refining recommendations: {e}")
        return []

# Function to register a new user
def register_user(username, password):
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb+srv://abhay21csu439:gangala12@recommendation.hoyhukz.mongodb.net/")
        db = client.get_database("recommendation-system")

        # Check if the username already exists
        if db.users.find_one({"username": username}):
            return False, "Username already exists"

        # Hash the password
        hashed_password = bcrypt.hash(password)

        # Insert user data into the users collection
        db.users.insert_one({"username": username, "password": hashed_password})

        return True, "Registration successful"
    except Exception as e:
        print(f"Error registering user: {e}")
        return False, "Registration failed"

# Function to authenticate a user
def authenticate_user(username, password):
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb+srv://abhay21csu439:gangala12@recommendation.hoyhukz.mongodb.net/")
        db = client.get_database("recommendation-system")

        # Retrieve user data from the users collection
        user_data = db.users.find_one({"username": username})

        # Check if the user exists and the password is correct
        if user_data and bcrypt.verify(password, user_data["password"]):
            return True, "Authentication successful"

        return False, "Invalid username or password"
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return False, "Authentication failed"

# Function to compute image embeddings using a pre-trained ResNet50 model
def get_embedding(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return model.predict(img_array).flatten()
# Function to compute color similarity between two embeddings
def compute_color_similarity(embedding1, embedding2):
    try:
        # Compute cosine similarity between embeddings
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    except Exception as e:
        print(f"Error computing color similarity: {e}")
        return None
    
    # Function to compute color similarity between two embeddings
def compute_color_similarity(embedding1, embedding2):
    try:
        # Compute cosine similarity between embeddings
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    except Exception as e:
        print(f"Error computing color similarity: {e}")
        return None

# Function to compute pattern similarity (using ORB feature detection)
def compute_pattern_similarity(img1, img2):
    try:
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initialize ORB detector
        orb = cv2.ORB_create()
        
        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        # Initialize matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = bf.match(des1, des2)
        
        # Compute pattern similarity (using number of matches)
        similarity = len(matches)
        
        return similarity
    except Exception as e:
        print(f"Error computing pattern similarity: {e}")
        return None


from sklearn.metrics.pairwise import cosine_similarity

# Function to recommend related products based on category, color, and pattern similarity
def recommend_related_products(selected_embedding, selected_category, all_embeddings, all_image_urls, top_n=5):
    try:
        # Initialize lists to store similarity scores
        combined_scores = []

        # Compute cosine similarity for color and pattern similarities
        color_similarities = cosine_similarity(selected_embedding.reshape(1, -1), all_embeddings)
        pattern_similarities = cosine_similarity(selected_embedding.reshape(1, -1), all_embeddings)

        # Combine category, color, and pattern similarities into a single score
        for cat_sim, color_sim, pattern_sim in zip(selected_category, color_similarities[0], pattern_similarities[0]):
            combined_score = 0.5 * cat_sim + 0.3 * color_sim + 0.2 * pattern_sim  # Adjust weights as needed
            combined_scores.append(combined_score)

        # Sort products based on combined score in descending order
        sorted_indices = np.argsort(combined_scores)[::-1]

        # Select top N most similar products
        related_products = [(all_image_urls[idx], combined_scores[idx]) for idx in sorted_indices[:top_n]]

        # Sort related products based on similarity score in descending order
        related_products.sort(key=lambda x: x[1], reverse=True)

        return related_products
    except Exception as e:
        print(f"Error recommending related products: {e}")
        return []


# Function to display the main UI
def main():
    st.title("Shopping Recommendation System")

    # Check if the user is logged in
    if st.session_state.get("logged_in"):
        # User is logged in
        st.write("Welcome, " + st.session_state.username)

        # Sidebar for category selection
        main_category = st.sidebar.selectbox("Select Main Category", ["Men", "Women"])

        # Subcategory and article type selection
        if main_category:
            subcategory = st.sidebar.selectbox("Select Subcategory", ["Apparel", "Footwear", "Accessories"])
            if subcategory:
                article_types = []
                # Dynamically fetch article types based on selected subcategory
                if subcategory == "Apparel":
                    article_types = ["Shirts", "Trousers", "Jeans", "T-Shirts"]
                elif subcategory == "Footwear":
                    article_types = ["Shoes", "Sneakers", "Sandals", "Boots"]
                elif subcategory == "Accessories":
                    article_types = ["Watches", "Bags", "Belts", "Hats"]
                article_type = st.sidebar.selectbox("Select Article Type", ["All"] + article_types)

        # User Input
        keyword = st.text_input("Enter product category or keywords:")

        if keyword:
            # Fetch image URLs from MongoDB based on the entered keyword
            st.write(f"Fetching image URLs for '{keyword}' from MongoDB...")
            image_urls = fetch_image_urls(keyword)

            if image_urls:
                # Display all the images
                st.write("Recommended Products:")
                for url in image_urls:
                    st.image(url, width=200, caption="Click to view details")

                    # Handle click events with buttons
                    if st.button("View Details", key=url):
                        # If the user clicks on the button, fetch and display details for that product
                        selected_product_details = fetch_image_details(url)
                        if selected_product_details:
                            st.write("Details:")
                            st.write(f"Description: {selected_product_details.get('productDisplayName', 'N/A')}")
                            st.write(f"Price: {selected_product_details.get('price', 'N/A')}")
                            st.write(f"Brand: {selected_product_details.get('brand', 'N/A')}")
                            # Add more details as needed

                            # Compute image embedding for the selected product
                            selected_embedding = compute_image_embedding(url)

                            # Compute image embeddings for all other products
                            all_image_urls = image_urls.copy()
                            all_image_urls.remove(url)
                            all_embeddings = np.array([compute_image_embedding(image_url) for image_url in all_image_urls])

                            # Recommend related products based on color similarity
                            st.write("Related Products Based on Color Similarity:")
                            selected_category = "Your selected category"
                            related_products = recommend_related_products(selected_embedding, selected_category, all_embeddings, all_image_urls)

                            for related_product in related_products:
                                st.image(related_product[0], width=200, caption=f"Similarity: {float(related_product[1]):.2f}")

                            # Refine recommendations based on similarity in design
                            st.write("Refining Recommendations Based on Design Similarity:")
                            refined_recommendations = refine_recommendations(selected_embedding, all_embeddings, all_image_urls)
                            for refined_product in refined_recommendations:
                                st.image(refined_product[0], width=200, caption=f"Similarity: {float(refined_product[1]):.2f}")

                            break  # Stop iterating once the details for the selected product are displayed

            else:
                st.error(f"No images available for the entered category or keywords: {keyword}")

        # Logout button
        if st.button("Search"):
            st.session_state.clear()
            st.write("Logged out successfully")

    else:
        # User is not logged in
        st.write("Please log in to continue")

        # Login Form
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            authenticated, message = authenticate_user(username, password)
            if authenticated:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.write("Login successful")
            else:
                st.error(message)

        # Registration Form
        st.write("New User? Register here:")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            registered, message = register_user(new_username, new_password)
            if registered:
                st.session_state.logged_in = True
                st.session_state.username = new_username
                st.write(message)
            else:
                st.error(message)

if __name__ == "__main__":
    main()
