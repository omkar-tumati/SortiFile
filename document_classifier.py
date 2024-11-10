import torch
from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification
from PIL import Image
import pytesseract
from torchvision import transforms
import cv2
import numpy as np
import os

class DocumentClassifier:
    def __init__(self):
        # Initialize LayoutLM model and tokenizer
        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.model = LayoutLMForSequenceClassification.from_pretrained(
            "microsoft/layoutlm-base-uncased",
            num_labels=len(self.get_document_types())
        )
        
        # Define image transformations
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_document_types():
        return {
            0: "Invoice",
            1: "Resume",
            2: "Contract",
            3: "ID Card",
            4: "Bank Statement"
        }

    def preprocess_image(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        # Get bounding boxes for text
        boxes = pytesseract.image_to_boxes(image)
        
        # Convert boxes to normalized format
        normalized_boxes = []
        words = []
        
        for box in boxes.splitlines():
            b = box.split()
            x1 = int(b[1]) / width * 1000  # Normalize x1
            y1 = (height - int(b[2])) / height * 1000  # Normalize y1
            x2 = int(b[3]) / width * 1000  # Normalize x2
            y2 = (height - int(b[4])) / height * 1000  # Normalize y2
            normalized_boxes.append([x1, y1, x2, y2])
            words.append(b[0])
        
        # Convert to model inputs
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Ensure bbox tensor is same length as tokenized input
        num_boxes = len(normalized_boxes)
        input_length = encoding['input_ids'].size(1)
        
        if num_boxes < input_length:
            # If there are fewer boxes than tokens, pad the boxes
            normalized_boxes += [[0, 0, 0, 0]] * (input_length - num_boxes)
        elif num_boxes > input_length:
            # If there are more boxes than tokens, truncate the boxes
            normalized_boxes = normalized_boxes[:input_length]
            
        # Convert normalized boxes to tensor
        encoding['bbox'] = torch.tensor([normalized_boxes], dtype=torch.float)
        
        return encoding



    def detect_logo(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply logo detection techniques
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        potential_logos = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:  # Common logo aspect ratio range
                    potential_logos.append((x, y, w, h))
        
        return len(potential_logos) > 0

    def classify_document(self, image_path):
        try:
            # Preprocess image and extract features
            encoding = self.preprocess_image(image_path)
            
            # Detect logo presence
            has_logo = self.detect_logo(image_path)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = outputs.logits.softmax(dim=1)
                
            # Get predicted class
            predicted_class = predictions.argmax().item()
            confidence = predictions.max().item()
            
            # Adjust prediction based on logo presence
            if has_logo and predicted_class in [0, 2]:  # Invoice or Contract
                confidence += 0.1  # Boost confidence if logo is present
                
            return {
                'document_type': self.get_document_types()[predicted_class],
                'confidence': confidence,
                'has_logo': has_logo
            }
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return None

# Example usage
def main():
    # Initialize classifier
    classifier = DocumentClassifier()
    
    # Example document classification
    image_path = "/Users/internalis/Documents/SortiFile/Data/invoive1.png"  # Replace with your document path
    try:
        result = classifier.classify_document(image_path)
        if result:
            print(f"Document Type: {result['document_type']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Logo Detected: {'Yes' if result['has_logo'] else 'No'}")
    except Exception as e:
        print(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main()
