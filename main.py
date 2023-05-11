import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from ttkthemes import ThemedStyle
from UI import classification_window
import PyPDF2


def digitization_window():
    main_frame.pack_forget()
    digitization_frame.pack()

    # new_window = tk.Toplevel()
    # new_window.title("Upload Image")
    # new_window.configure(bg="#ADCFCD")

    # new_window.geometry("200x200")

    # upload_button = tk.Button(new_window, text="Upload Image", font=(
    #     "Arial", 11), command=upload_file, width=10, height=1)
    # upload_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # label1 = tk.Label(new_window, text="Upload a PDF file only",
    #                   foreground="red", font=("Arial", 10), bg="#ADCFCD")
    # label1.place(relx=0.5, rely=0.65, anchor=tk.CENTER)


def go_back():
    digitization_frame.pack_forget()
    main_frame.pack()


def upload_ecg_image():
    file_path = filedialog.askopenfilename(title="Select ECG Image", filetypes=(
        ("PDF Files", "*.pdf"), ("All Files", "*.*")))
    if file_path:
        try:
            pdf_file = open(file_path, 'rb')
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Assuming there's only one page in the PDF file
            page = pdf_reader.pages[0]
            # image = page.extract_images()[0]['image']
            display_image(page)

        except Exception as e:
            messagebox.showerror("Error", str(e))


def display_image(image):
    image = image.resize((300, 400))
    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)
    image_label.image = photo
    digitize_button.pack()


def digitize_ecg():
    messagebox.showinfo("Digitize", "ECG Digitized!")


# Create the main window
root = tk.Tk()
root.title("ECG Digitization and Classification")

# Create the main frame
main_frame = tk.Frame(root)
main_frame.pack(pady=20)
# Add the main heading label

heading_label = tk.Label(
    main_frame, text="ECG Digitization and Classification", font=("Arial", 20))
heading_label.pack()
heading_label.pack()

# Create a frame for the buttons
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=50)

# Load images for the buttons
digitization_image = Image.open("digitization.png")
classification_image = Image.open("classification.png")

# Resize the images to fit the buttons
digitization_image = digitization_image.resize((150, 150))
classification_image = classification_image.resize((150, 150))

# Create PhotoImage objects from the resized images
digitization_photo = ImageTk.PhotoImage(digitization_image)
classification_photo = ImageTk.PhotoImage(classification_image)

# Create the digitization button with the image and label
digitization_button = tk.Button(button_frame, image=digitization_photo, text="Digitization",
                                compound=tk.TOP, command=digitization_window)
digitization_button.pack(side=tk.LEFT, padx=20)

# Create the classification button with the image and label
classification_button = tk.Button(button_frame, image=classification_photo, text="Classification",
                                  compound=tk.TOP, command=classification_window)
classification_button.pack(side=tk.LEFT, padx=20)


# Create a frame for the digitization screen
digitization_frame = tk.Frame(root)

# Add the back button to the digitization screen
back_button = tk.Button(digitization_frame, text="Back", command=go_back)
back_button.pack(anchor=tk.NW)

# Add the upload ECG image button to the digitization screen
upload_button = tk.Button(
    digitization_frame, text="Upload ECG Image", command=upload_ecg_image)
upload_button.pack()

# Create a label to display the uploaded PDF image
image_label = tk.Label(digitization_frame)
image_label.pack(pady=10)

# Create the digitize button
digitize_button = tk.Button(
    digitization_frame, text="Digitize", command=digitize_ecg)

# Initially hide the digitize button
digitize_button.pack_forget()

# Switch to the main screen initially
go_back()

# Start the main event loop
root.mainloop()
