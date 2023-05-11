import tkinter as tk
from tkinter import filedialog
from tkinter import ttk


def upload_file():
    file_path = filedialog.askopenfilename(
        title="Select file", filetypes=(("Files", "*.pdf"), ("All Files", "*")))
    if file_path:

        # Create a new window to display the PDF file
        new_window = tk.Toplevel()
        new_window.title("Classify ECG")
        new_window.configure(bg="#ADCFCD")

        new_window.geometry("200x200")

        classify_button = tk.Button(new_window, text="Digitize ECG")
        classify_button.pack()

    # Create a scrollbar widget and connect it to the canvas


def upload_csv():
    file_path = filedialog.askopenfilename(
        title="Select file", filetypes=(("Files", "*.csv"), ("All Files", "*")))

    if file_path:
        # Create a new window to display the PDF file
        new_window = tk.Toplevel()
        new_window.title("Classify ECG")
        new_window.configure(bg="#ADCFCD")

        new_window.geometry("200x200")

        classify_button = tk.Button(new_window, text="Classify ECG")
        classify_button.pack()


def classification_window():
    new_window = tk.Toplevel()
    new_window.title("Upload File")
    new_window.configure(bg="#ADCFCD")

    new_window.geometry("200x200")

    upload_button = tk.Button(new_window, text="Upload File", font=("Arial", 11), command=upload_csv, width=10,
                              height=1)
    upload_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    label1 = tk.Label(new_window, text="Upload CSV file only",
                      foreground="red", font=("Arial", 10), bg="#ADCFCD")
    label1.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
