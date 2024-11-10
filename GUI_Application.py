"""
Modern Apple-Style Document Classifier GUI
Requirements: Python 3.x, tkinter, requests, ttkthemes
Install requirements:
pip install requests ttkthemes
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ttkthemes import ThemedTk
import requests
import base64
import json
import os
import csv
from datetime import datetime
from pathlib import Path
import shutil

class ModernDocumentClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Classifier")
        self.root.geometry("900x700")
        
        # Set theme and configure colors
        self.configure_theme()
        
        # Initialize output directories
        self.setup_directories()
        
        # Create main container with padding
        self.main_container = ttk.Frame(root, padding="20")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)
        
        self.create_gui_elements()
        
        # Initialize results storage
        self.current_results = {}

    def configure_theme(self):
        # Configure colors and styles
        style = ttk.Style()
        
        # Configure main elements
        style.configure("Main.TFrame", background="#f5f5f7")
        style.configure("Card.TFrame", background="white")
        
        # Configure buttons
        style.configure("Accent.TButton",
                       padding=10,
                       font=("SF Pro Text", 11))
        
        # Configure labels
        style.configure("Header.TLabel",
                       font=("SF Pro Display", 16, "bold"),
                       background="#f5f5f7")
        style.configure("SubHeader.TLabel",
                       font=("SF Pro Text", 12),
                       background="#f5f5f7")
        
        # Configure treeview
        style.configure("Treeview",
                       font=("SF Pro Text", 11),
                       rowheight=30)
        style.configure("Treeview.Heading",
                       font=("SF Pro Text", 12, "bold"))

    def setup_directories(self):
        """Setup output directory structure"""
        self.base_dir = Path("classification_results")
        self.json_dir = self.base_dir / "json"
        self.text_dir = self.base_dir / "text"
        self.archive_dir = self.base_dir / "archive"
        
        # Create directories if they don't exist
        for directory in [self.base_dir, self.json_dir, self.text_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize or load the CSV log file
        self.log_file = self.base_dir / "classification_log.csv"
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Filename', 'Document Type', 'Confidence', 'Path'])

    def create_gui_elements(self):
        # Header section
        header_frame = ttk.Frame(self.main_container, style="Main.TFrame")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        
        header_label = ttk.Label(header_frame,
                               text="Document Classifier",
                               style="Header.TLabel")
        header_label.grid(row=0, column=0, sticky="w")
        
        # Input section
        input_frame = ttk.Frame(self.main_container, style="Card.TFrame")
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        
        # Model selection
        model_label = ttk.Label(input_frame,
                              text="Model:",
                              style="SubHeader.TLabel")
        model_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.model_var = tk.StringVar(value="x/llama3.2-vision:latest")
        model_entry = ttk.Entry(input_frame,
                              textvariable=self.model_var,
                              width=40)
        model_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # File selection
        file_label = ttk.Label(input_frame,
                             text="File:",
                             style="SubHeader.TLabel")
        file_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(input_frame,
                             textvariable=self.file_path,
                             width=40)
        file_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        
        browse_button = ttk.Button(input_frame,
                                 text="Browse",
                                 style="Accent.TButton",
                                 command=self.browse_file)
        browse_button.grid(row=1, column=2, padx=10, pady=10)
        
        # Classify button
        classify_button = ttk.Button(input_frame,
                                   text="Classify Document",
                                   style="Accent.TButton",
                                   command=self.classify_and_save)
        classify_button.grid(row=2, column=0, columnspan=3, pady=20)
        
        # Results section
        self.create_results_section()
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.main_container,
                             textvariable=self.status_var,
                             style="SubHeader.TLabel")
        status_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def create_results_section(self):
        # Create notebook for results
        self.results_notebook = ttk.Notebook(self.main_container)
        self.results_notebook.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 20))
        
        # Results tab
        results_frame = ttk.Frame(self.results_notebook, style="Card.TFrame")
        self.results_notebook.add(results_frame, text="Results")
        
        self.result_text = tk.Text(results_frame,
                                 height=15,
                                 width=70,
                                 font=("SF Pro Text", 11),
                                 wrap=tk.WORD)
        self.result_text.pack(expand=True, fill="both", padx=10, pady=10)
        
        # JSON tab
        json_frame = ttk.Frame(self.results_notebook, style="Card.TFrame")
        self.results_notebook.add(json_frame, text="JSON")
        
        self.json_text = tk.Text(json_frame,
                               height=15,
                               width=70,
                               font=("SF Mono", 11),
                               wrap=tk.WORD)
        self.json_text.pack(expand=True, fill="both", padx=10, pady=10)
        
        # History tab
        history_frame = ttk.Frame(self.results_notebook, style="Card.TFrame")
        self.results_notebook.add(history_frame, text="History")
        
        self.history_tree = ttk.Treeview(history_frame,
                                       columns=('Time', 'Type', 'File'),
                                       show='headings',
                                       style="Treeview")
        self.history_tree.heading('Time', text='Time')
        self.history_tree.heading('Type', text='Document Type')
        self.history_tree.heading('File', text='Filename')
        
        # Configure column widths
        self.history_tree.column('Time', width=150)
        self.history_tree.column('Type', width=200)
        self.history_tree.column('File', width=300)
        
        self.history_tree.pack(expand=True, fill="both", padx=10, pady=10)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Document",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("All files", "*.*")
            )
        )
        if filename:
            self.file_path.set(filename)

    def classify_document(self, image_path):
        """Document classification with API call"""
        OLLAMA_API_URL = "http://localhost:11434/api/generate"
        
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            
            data = {
                "model": self.model_var.get(),
                "prompt": "Classify the document type. Return only the category in JSON format.",
                "images": [image_base64],
                "format": "json",
                "stream": False
            }
            
            response = requests.post(OLLAMA_API_URL, json=data)
            response_json = response.json()
            
            if response_json.get("done"):
                output_json = response_json.get("response")
                if output_json:
                    try:
                        result = eval(output_json)
                        # Enhance result with additional metadata
                        return {
                            "category": result.get("category", "Unknown"),
                            "confidence": 0.95,  # Replace with actual confidence if available
                            "details": {
                                "processed_time": datetime.now().isoformat(),
                                "model_version": self.model_var.get(),
                                "file_info": {
                                    "name": os.path.basename(image_path),
                                    "size": os.path.getsize(image_path),
                                    "path": image_path
                                }
                            }
                        }
                    except (SyntaxError, KeyError) as e:
                        raise Exception(f"Error parsing API response: {str(e)}")
            
            raise Exception("Classification failed")
            
        except Exception as e:
            return {"error": str(e)}

    def save_results(self, results, image_path):
        """Save results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{timestamp}_{Path(image_path).stem}"
        
        # Save JSON results
        json_path = self.json_dir / f"{filename_base}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save text summary
        text_path = self.text_dir / f"{filename_base}.txt"
        with open(text_path, 'w') as f:
            f.write(f"Document Classification Results\n")
            f.write(f"===========================\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Document Type: {results.get('category', 'Unknown')}\n")
            f.write(f"Confidence: {results.get('confidence', 0):.2%}\n")
            f.write(f"File: {Path(image_path).name}\n")
            f.write(f"Path: {image_path}\n")
            
        # Update CSV log
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                Path(image_path).name,
                results.get('category', 'Unknown'),
                f"{results.get('confidence', 0):.2%}",
                image_path
            ])
            
        return text_path, json_path

    def update_display(self, results, text_path, json_path):
        """Update all display elements with new results"""
        # Update text result display
        self.result_text.delete(1.0, tk.END)
        with open(text_path, 'r') as f:
            self.result_text.insert(tk.END, f.read())
            
        # Update JSON display
        self.json_text.delete(1.0, tk.END)
        self.json_text.insert(tk.END, json.dumps(results, indent=4))
        
        # Update history
        self.history_tree.insert('', 0, values=(
            datetime.now().strftime('%H:%M:%S'),
            results.get('category', 'Unknown'),
            Path(results['details']['file_info']['path']).name
        ))

    def classify_and_save(self):
        """Enhanced classification and save workflow"""
        if not self.file_path.get():
            messagebox.showwarning("Warning", "Please select a file first")
            return
            
        self.status_var.set("Processing...")
        self.root.update()
        
        try:
            # Get classification results
            results = self.classify_document(self.file_path.get())
            
            if "error" in results:
                raise Exception(results["error"])
            
            # Save results
            text_path, json_path = self.save_results(results, self.file_path.get())
            
            # Update display
            self.update_display(results, text_path, json_path)
            
            self.status_var.set("Results saved successfully")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Classification failed: {str(e)}")

def main():
    root = ThemedTk(theme="arc")  # Using arc theme for clean, modern look
    app = ModernDocumentClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()