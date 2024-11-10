import unittest
from document_classifier import DocumentClassifier
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil

class TestDocumentClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and create sample documents"""
        cls.classifier = DocumentClassifier()
        cls.test_dir = "test_documents"
        
        # Create test directory if it doesn't exist
        if not os.path.exists(cls.test_dir):
            os.makedirs(cls.test_dir)
        
        # Create sample documents
        cls.create_sample_documents()

    @classmethod
    def create_sample_documents(cls):
        """Create sample documents for testing"""
        # Sample document templates
        documents = {
            'invoice': {
                'title': 'INVOICE',
                'content': [
                    'Invoice #: INV-2024-001',
                    'Date: 04/11/2024',
                    'Bill To:',
                    'John Doe',
                    'Amount Due: $1,500.00',
                    'Due Date: 04/25/2024'
                ],
                'has_logo': True
            },
            'resume': {
                'title': 'PROFESSIONAL RESUME',
                'content': [
                    'John Smith',
                    'Software Engineer',
                    'Experience:',
                    'Senior Developer - Tech Corp',
                    'Education:',
                    'BS Computer Science'
                ],
                'has_logo': False
            },
            'contract': {
                'title': 'SERVICE CONTRACT',
                'content': [
                    'AGREEMENT',
                    'Between: Company A',
                    'And: Company B',
                    'Terms and Conditions:',
                    '1. Service Description',
                    '2. Payment Terms'
                ],
                'has_logo': True
            }
        }

        for doc_type, content in documents.items():
            # Create a new image
            img = Image.new('RGB', (1000, 1400), color='white')
            d = ImageDraw.Draw(img)
            
            try:
                # Use a basic font
                font_title = ImageFont.truetype("arial.ttf", 36)
                font_body = ImageFont.truetype("arial.ttf", 24)
            except:
                # Fallback to default font if arial is not available
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()

            # Add title
            d.text((100, 50), content['title'], fill='black', font=font_title)
            
            # Add content
            y_position = 150
            for line in content['content']:
                d.text((100, y_position), line, fill='black', font=font_body)
                y_position += 50

            # Add simple logo if required
            if content['has_logo']:
                logo_box = [(50, 50), (150, 150)]
                d.rectangle(logo_box, outline='black', fill='gray')

            # Save the document
            filename = f"{cls.test_dir}/{doc_type}_sample.png"
            img.save(filename)
            print(f"Created test document: {filename}")

    def test_document_types(self):
        """Test if the classifier can identify different document types"""
        test_files = {
            f"{self.test_dir}/invoice_sample.png": "Invoice",
            f"{self.test_dir}/resume_sample.png": "Resume",
            f"{self.test_dir}/contract_sample.png": "Contract"
        }

        for file_path, expected_type in test_files.items():
            with self.subTest(file_path=file_path):
                result = self.classifier.classify_document(file_path)
                print(f"\nTesting {file_path}:")
                if result is None:
                    self.fail(f"No result returned for {file_path}")

                print(f"Predicted Type: {result['document_type']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Logo Detected: {result['has_logo']}")
                self.assertEqual(result['document_type'], expected_type)

    def test_logo_detection(self):
        """Test logo detection functionality"""
        # Test files with logos
        files_with_logos = [
            f"{self.test_dir}/invoice_sample.png",
            f"{self.test_dir}/contract_sample.png"
        ]
        
        # Test files without logos
        files_without_logos = [
            f"{self.test_dir}/resume_sample.png"
        ]

        # Test files with logos
        for file_path in files_with_logos:
            with self.subTest(file_path=file_path):
                has_logo = self.classifier.detect_logo(file_path)
                self.assertTrue(has_logo, f"Logo not detected in {file_path}")

        # Test files without logos
        for file_path in files_without_logos:
            with self.subTest(file_path=file_path):
                has_logo = self.classifier.detect_logo(file_path)
                self.assertFalse(has_logo, f"Logo falsely detected in {file_path}")

    def test_confidence_scores(self):
        """Test if confidence scores are within expected range"""
        test_files = os.listdir(self.test_dir)
        for file_name in test_files:
            file_path = os.path.join(self.test_dir, file_name)
            with self.subTest(file_path=file_path):
                result = self.classifier.classify_document(file_path)
                if result is None:
                    self.fail(f"No result returned for {file_path}")

                self.assertGreaterEqual(result['confidence'], 0.0)
                self.assertLessEqual(result['confidence'], 1.1)  # 1.1 to account for logo boost

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove test documents
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

def run_tests():
    """Run the test suite"""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDocumentClassifier)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests()
