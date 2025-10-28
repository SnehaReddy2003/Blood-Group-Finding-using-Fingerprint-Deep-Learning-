from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = load_model("blood_group_model.h5")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Fingerprint verification function
def is_fingerprint(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        # Edge metrics
        edge_count = np.sum(edges > 0)
        total_pixels = edges.size
        edge_percentage = (edge_count / total_pixels) * 100

        # Very relaxed thresholds
        min_edge_percent = 2.5  # Lowered from 3.5
        max_edge_percent = 30.0  # Raised from 25.0
        threshold = total_pixels * (min_edge_percent / 100)

        # Variance and ridge frequency
        variance = np.var(gray)
        roi = gray[gray.shape[0]//4:3*gray.shape[0]//4, gray.shape[1]//4:3*gray.shape[1]//4]
        freq = np.fft.fft2(roi)
        freq_power = np.abs(freq) ** 2
        ridge_freq = np.mean(freq_power) > 500  # Lowered from 700

        # Detailed logging
        logger.debug(f"Edge count: {edge_count}, Total pixels: {total_pixels}")
        logger.debug(f"Edge percentage: {edge_percentage:.2f}% (Range: {min_edge_percent}-{max_edge_percent})")
        logger.debug(f"Variance: {variance:.2f} (Min: 50)")
        logger.debug(f"Ridge freq power: {np.mean(freq_power):.2f} (Min: 500)")

        # Fingerprint criteria (very relaxed)
        is_fp = (edge_count > threshold and 
                 min_edge_percent <= edge_percentage <= max_edge_percent and 
                 variance > 50 and  # Lowered from 70
                 ridge_freq)

        # Log rejection reasons
        if not is_fp:
            if edge_count <= threshold:
                logger.debug("Rejected: Insufficient edges")
            elif not (min_edge_percent <= edge_percentage <= max_edge_percent):
                logger.debug(f"Rejected: Edge percentage {edge_percentage:.2f} out of range")
            elif variance <= 50:
                logger.debug(f"Rejected: Variance too low ({variance:.2f})")
            elif not ridge_freq:
                logger.debug("Rejected: No ridge frequency detected")

        logger.debug(f"Is fingerprint? {is_fp}")
        return is_fp
    except Exception as e:
        logger.error(f"Error in is_fingerprint: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict_blood_group():
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        # Read file data
        file_data = file.read()
        logger.info(f"Uploaded file size: {len(file_data)} bytes")
        
        # Load image in RGB for fingerprint check
        img_rgb = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if img_rgb is None:
            logger.error("Failed to decode image in RGB")
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400
        logger.debug(f"RGB image loaded: {img_rgb.shape}")
        
        # Verify fingerprint
        if not is_fingerprint(img_rgb):
            logger.warning("Image rejected: Not a fingerprint")
            return jsonify({'error': 'Uploaded image is not a fingerprint'}), 400
        logger.info("Fingerprint verification passed")

        # Reload image in grayscale for prediction
        img_gray = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            logger.error("Failed to decode image in grayscale")
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400
        logger.debug(f"Grayscale image loaded: {img_gray.shape}")
        
        # Preprocess image
        img = cv2.resize(img_gray, (128, 128))
        img = img.reshape(1, 128, 128, 1) / 255.0
        logger.debug(f"Preprocessed image shape: {img.shape}")
        
        # Predict
        prediction = model.predict(img)
        confidence = np.max(prediction)
        blood_group = blood_groups[np.argmax(prediction)]
        logger.debug(f"Prediction probabilities: {prediction[0]}")
        logger.info(f"Predicted blood group: {blood_group}, Confidence: {confidence:.4f}")
        
        logger.info(f"Returning blood group: {blood_group}")
        return jsonify({'blood_group': blood_group})
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)