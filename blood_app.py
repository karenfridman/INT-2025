import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#===============
# App config
#===============
st.set_page_config(page_title="Blood Cell Classifier", layout="wide")
st.title("🩸 Deep Cell - AI-powered blod cell reconition")
st.write("Upload an image of a blood cell (lymphocyte, monocythe, neutrophil or eosinophil), and the model will predict its type.")

#===============
# Load trained model
#===============
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_4class_blood_model_imp.keras", 
        compile=False
    )
    return model

model = load_model()

# Class names (adjust if different)
CLASS_NAMES = ['neutrophil', 'eosinophil', 'lymphocyte', 'monocyte']

#===============
# Preprocess images
#===============
# Cell cropping function
def extract_cell_from_nucleus(img):
    
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect purple nucleus
    lower_purple = np.array([110, 40, 40])
    upper_purple = np.array([170, 255, 255])
    nucleus_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Clean small specks
    kernel = np.ones((3, 3), np.uint8)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Grow outward to approximate full cell
    grow_mask = nucleus_mask.copy()
    for i in range(25):
        grow_mask = cv2.dilate(grow_mask, kernel, iterations=1)
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 30, 100)
        grow_mask[edges > 0] = 0  # stop at boundaries

    # Find largest contour (the cell)
    contours, _ = cv2.findContours(grow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c)
    cell_crop = img[y:y+h_box, x:x+w_box]
    return cell_crop

# Background removal
def preprocess_cell_keep_cytoplasm(img):
    img = tf.cast(img, tf.uint8)
    img_np = img.numpy()

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Nucleus (purple-blue) range
    lower_nucleus = np.array([110, 40, 40])
    upper_nucleus = np.array([170, 255, 255])
    nucleus_mask = cv2.inRange(hsv, lower_nucleus, upper_nucleus)

    # Cytoplasm (light pink-violet)
    lower_cytoplasm = np.array([140, 15, 90])
    upper_cytoplasm = np.array([179, 120, 255])
    cytoplasm_mask = cv2.inRange(hsv, lower_cytoplasm, upper_cytoplasm)

    # Combine
    combined_mask = cv2.bitwise_or(nucleus_mask, cytoplasm_mask)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest contour (the cell)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(combined_mask)
        cv2.drawContours(final_mask, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = combined_mask

    # Smooth edges
    final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

    # Apply mask to image
    result = cv2.bitwise_and(img_np, img_np, mask=final_mask)

    return result
def preprocess_image(image):
    result = extract_cell_from_nucleus(image)
    result = preprocess_cell_keep_cytoplasm(result)
    return result

#===============
# Grad-CAM function
#===============
def make_gradcam_heatmap(pre_img_batch, model, last_conv_layer_name="block7a_project_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(pre_img_batch)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        if len(predictions.shape) == 1:
            predictions = tf.expand_dims(predictions, axis=0)

        class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Safe conversion to numpy / int
    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()
    if isinstance(class_idx, tf.Tensor):
        class_idx = int(class_idx.numpy())
    else:
        class_idx = int(class_idx)

    return heatmap, class_idx

# ---- Function to Overlay Heatmap on Image ----
def overlay_gradcam(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(heatmap, alpha, original_img, 1 - alpha, 0)
    return overlay



#===============
# Image uploader
#===============
uploaded_file = st.file_uploader("Upload a blood cell image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Open and convert image
    pil_image = Image.open(uploaded_file).convert("RGB")
    
    # Resize to model input size
    pil_image = pil_image.resize((224, 224))
    
    # Convert to numpy array
    image = np.array(pil_image)
    
    # Convert RGB -> BGR if your model was trained with OpenCV format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Preprocess for model (normalize to [0,1])
    #pre_img = image / 255.0
    # Preprocess
    pre_img = preprocess_image(image)
    # Add batch dimension
    pre_img_batch = np.expand_dims(pre_img, axis=0)

    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    

#===============
#Prediction
#===============
    preds = model.predict(pre_img_batch)
    probs = tf.nn.softmax(preds[0]).numpy()

    predicted_class = CLASS_NAMES[np.argmax(probs)]
    confidence = np.max(probs) * 100

    with col2:
        st.metric("Predicted Cell Type", f"{predicted_class}", f"{confidence:.2f}% confidence")

#===============
# Probability chart
#===============
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, probs, color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    ax.set_title("Class Probabilities")
    st.pyplot(fig)

#===============
# Grad-CAM Visualization
#===============
    st.subheader("Grad-CAM Heatmap (Explainability)")
    # Generate Grad-CAM
    last_conv_layer_name = "block7a_project_conv"  # for EfficientNetB0
    heatmap, class_idx = make_gradcam_heatmap(pre_img_batch, model, last_conv_layer_name)

    # Overlay Grad-CAM on the original image
    cam_image = overlay_gradcam(image, heatmap)

    # Show results in Streamlit
    st.image(cam_image, channels="BGR", caption=f"Grad-CAM for class {class_idx}")

    st.markdown("### Image Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.image(pre_img, caption="Preprocessed Image (Model Input)", use_column_width=True)
    
    with col3:
        st.image(cam_image, caption=f"Grad-CAM Overlay (Class {class_idx})", use_column_width=True)

else:
    st.info("👆 Upload a blood cell image to start classification.")

st.markdown("---")
st.markdown(
    """
    <p style="text-align: center; color: gray; font-size: 0.9em;">
        Karen Fridman - INT Data Science Course Nov. 2025
    </p>
    """,
    unsafe_allow_html=True
)