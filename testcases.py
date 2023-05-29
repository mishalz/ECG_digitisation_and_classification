import unittest
import tkinter as tk
from PIL import Image, ImageTk
import main

class TestUI(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.title("Upload ECG Image")

    def tearDown(self):
        self.root.destroy()

    def test_heading_label(self):
        heading_text = "ECG Digitization And Classification"
        heading = tk.Label(self.root, text=heading_text, font=("Arial", 20, "bold"), bg="#ADCFCD")
        heading.place(relx=0.49, rely=0.2, anchor=tk.CENTER)
        self.assertEqual(heading.cget("text"), heading_text)

    def test_digitization_button(self):
        photo_path = "/Users/ayesha/Desktop/ECG/digitization.png"
        photo = ImageTk.PhotoImage(Image.open(photo_path).resize((100, 100)))
        digitization_button = tk.Button(self.root, image=photo, command=main.digitization_window)
        digitization_button.place(relx=0.4, rely=0.47, anchor=tk.CENTER)
        self.assertIsInstance(digitization_button, tk.Button)
        # Add more assertions for the button as needed

    def test_classification_button(self):
        photo1_path = "/Users/ayesha/Desktop/ECG/classification.png"
        photo1 = ImageTk.PhotoImage(Image.open(photo1_path).resize((100, 100)))
        classification_button = tk.Button(self.root, image=photo1, command=main.classification_window)
        classification_button.place(relx=0.57, rely=0.47, anchor=tk.CENTER)
        self.assertIsInstance(classification_button, tk.Button)
        # Add more assertions for the button as needed

    def test_labels(self):
        dig_label_text = "Digitize ECG"
        classify_label_text = "Classify ECG"
        dig_label = tk.Label(self.root, text=dig_label_text, font=("Arial", 12, "bold"), bg="#ADCFCD")
        classify_label = tk.Label(self.root, text=classify_label_text, font=("Arial", 12, "bold"), bg="#ADCFCD")
        self.assertEqual(dig_label.cget("text"), dig_label_text)
        self.assertEqual(classify_label.cget("text"), classify_label_text)


class ClassificationWindowTestCase(unittest.TestCase):
  def test_classification_window(self):
        root = tk.Tk()
        main.classification_window()
        root.update()

        # Verify that a new window is created
        new_window = root.winfo_children()[0]
        self.assertIsInstance(new_window, tk.Toplevel)

        # Verify the title and background color of the new window
        self.assertEqual(new_window.title(), "Upload File")
        self.assertEqual(new_window.cget("bg"), "#ADCFCD")

        # Verify the window size
        self.assertEqual(new_window.winfo_width(), 200)
        self.assertEqual(new_window.winfo_height(), 200)

        # Verify the presence of the upload button and label
        upload_button = new_window.winfo_children()[0]
        label1 = new_window.winfo_children()[1]
        self.assertIsInstance(upload_button, tk.Button)
        self.assertIsInstance(label1, tk.Label)
        self.assertEqual(upload_button.cget("text"), "Upload File")
        self.assertEqual(label1.cget("text"), "Upload CSV file only")
        self.assertEqual(label1.cget("foreground"), "red")

        root.destroy()

class DigitizationWindowTestCase(unittest.TestCase):
    def test_digitization_window(self):
        root = tk.Tk()
        main.digitization_window()
        root.update()

        # Verify that a new window is created
        new_window = root.winfo_children()[0]
        self.assertIsInstance(new_window, tk.Toplevel)

        # Verify the title and background color of the new window
        self.assertEqual(new_window.title(), "Upload Image")
        self.assertEqual(new_window.cget("bg"), "#ADCFCD")

        # Verify the window size
        self.assertEqual(new_window.winfo_width(), 300)
        self.assertEqual(new_window.winfo_height(), 300)

        # Verify the presence of the upload button and label
        upload_button = new_window.winfo_children()[0]
        label1 = new_window.winfo_children()[1]
        self.assertIsInstance(upload_button, tk.Button)
        self.assertIsInstance(label1, tk.Label)
        self.assertEqual(upload_button.cget("text"), "Upload Image")
        self.assertEqual(label1.cget("text"), "Upload a PDF file only")
        self.assertEqual(label1.cget("foreground"), "red")

        root.destroy()

if __name__ == "__main__":
    unittest.main()
