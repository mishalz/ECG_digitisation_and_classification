from distutils import command
from email.mime import image
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from ttkthemes import ThemedStyle
#from UI import classification_window
from PyPDF2 import PdfReader
#from Preprocessing import image_preprocessing
#import fitz
#from pdf2image import convert_from_path
from SignalExtraction import getDigitisedSignal , drawECGSignal,downloadDigitisedSignal
import test_model
import load_model



'''Digitization'''

#global variables
uploaded_image = None
sorted_points=None

def digitization_window():
    main_frame.pack_forget()
    digitization_frame.pack()

    '''new_window = tk.Toplevel()
    new_window.title("Upload Image")
    new_window.configure(bg="#ADCFCD")

    new_window.geometry("200x200")

    upload_button = tk.Button(new_window, text="Upload Image", font=(
        "Arial", 11), command=upload_file, width=10, height=1)
    upload_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    label1 = tk.Label(new_window, text="Upload a PDF file only",
                     foreground="red", font=("Arial", 10), bg="#ADCFCD")
    label1.place(relx=0.5, rely=0.65, anchor=tk.CENTER)'''

def classification_window():
    main_frame.pack_forget()
    classification_frame.pack()




def signal_window():
    digitize_ecg(uploaded_image)
    #digitization_frame.pack_forget()
    #upload_button.pack_forget()
    digitize_button.pack_forget()
    csv_button.pack()
    signal_frame.pack()

def go_back():
    csv_button.pack_forget()
    digitization_frame.pack_forget()
    signal_frame.pack_forget()
    main_frame.pack()


def upload_ecg_image():
    global uploaded_image
    file_path = filedialog.askopenfilename(title="Select ECG Image", filetypes=(
        ("JPEG Files", "*.jpg"), ("All Files", "*.*")))
    if file_path:
        try:
            # Convert the PDF to a list of PIL images
            #images = convert_from_path(file_path)
            #image = images[0]
            image = Image.open(file_path)
            display_image(image)
            uploaded_image = image
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

def display_image(image):
    image = image.resize((1200, 500))
    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)
    image_label.image = photo
    digitize_button.pack()


def digitize_ecg(image):
    global sorted_points
    #messagebox.showinfo("Digitize", "ECG Digitized!")
    #closed_image = image_preprocessing(image)
    sorted_points = getDigitisedSignal(image)
    digitized_signal= drawECGSignal(sorted_points)
    #image_object = Image.fromarray(digitized_signal)
    return digitized_signal

'''classification'''

def go_back1():
    classification_frame.pack_forget()
    main_frame.pack()

def go_back2():
    classification_frame2.pack_forget()
    classification_frame.pack()

def display_labels(actual_labels,predicted_labels): #display result of classsification
    classification_frame.pack_forget()
    labels=['/', 'A', 'F', 'J', 'L', 'N', 'Q','R','V','a','e','f','j']
    listbox.delete(0, tk.END)

    # Insert the actual and predicted labels into the listbox
    for i in range(len(actual_labels)):
        listbox.insert(tk.END, f"Actual Label: {actual_labels[i]}")
        decode = labels[predicted_labels[i]]
        listbox.insert(tk.END, f"Predicted Label: {decode}")
    classification_frame2.pack()
    back_button3.pack()


def upload_csv():
    file_path = filedialog.askopenfilename(
        title="Select file", filetypes=(("Files", "*.csv"), ("All Files", "*")))

    if file_path:
        try:

         # Create a label to display the success message
            #upload_csv.pack_forget()
            #success_label = tk.Label(classification_frame, text="CSV file uploaded successfully!")
            #success_label.pack()
    
    # Create a button for classification
            test = test_model
            labels, predictions = test.test_model.compute(test,file_path)
            x=0
            while(x<2):
               
                x+=1
               
            #classify_button = tk.Button(classification_frame, text="Classify", command=display_labels(labels,predictions))
            #classify_button.pack()
            display_labels(labels,predictions)
        except Exception as e:
            messagebox.showerror("Error", str(e))





# Create the main window
root = tk.Tk()
root.title("ECG Digitization and Classification")

# Create the main frame
main_frame = tk.Frame(root)
main_frame.pack(pady=50)
## Pack the frame and disable automatic resizing
main_frame.pack_propagate(0)
# Set the fixed size of the frame
main_frame.config(width=900, height=500)


# Add the main heading label
heading_label = tk.Label(
    main_frame, text="ECG DIGITIZATION AND CLASSIFICATION", font=("Arial", 22))
heading_label.pack()
heading_label.pack()

# Create a frame for the buttons
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=80)


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
digitization_button.pack(side=tk.LEFT, padx=40)

# Create the classification button with the image and label
classification_button = tk.Button(button_frame, image=classification_photo, text="Classification",
                                  compound=tk.TOP, command=classification_window)
classification_button.pack(side=tk.LEFT, padx=40)


# Create a frame for the digitization screen
digitization_frame = tk.Frame(root)

#create a frame for displaying the extracted signal 
signal_frame = tk.Frame(root)

classification_frame=tk.Frame(root,width=400, height=300)
classification_frame2=tk.Frame(root,width=400, height=300)

# Add the back button to the digitization screen
back_button = tk.Button(digitization_frame, text="Back", command=go_back)
back_button.pack(anchor=tk.NW)


back_button2 = tk.Button(classification_frame, text="Back", command=go_back1)
back_button2.pack(anchor=tk.NW)

back_button3 = tk.Button(classification_frame2, text="Back", command=go_back2)
back_button3.pack(anchor=tk.NW)


# Add the upload ECG image button to the digitization screen
upload_button = tk.Button(
    digitization_frame, text="Upload ECG Image", command=upload_ecg_image)
upload_button.pack()

# Create a label to display the uploaded PDF image
image_label = tk.Label(digitization_frame)
image_label.pack(pady=10)


# Create the digitize button
digitize_button = tk.Button(
    digitization_frame, text="Digitize", command=signal_window)

 # Load the saved image
plot_image = tk.PhotoImage(file='ecg_plot.png')

# Display the image on a Tkinter label widget
plot_label = tk.Label(signal_frame, image=plot_image)
plot_label.pack()

#_, sorted_points = digitize_ecg(uploaded_image)
#create the download csv file button 
csv_button = tk.Button(signal_frame , text="Download CSV file",command=lambda:downloadDigitisedSignal(sorted_points, 'a', 7))
csv_button.pack()


# Initially hide the digitize button
digitize_button.pack_forget()

upload_csv = tk.Button(
    classification_frame, text="Upload CSV", command=upload_csv)
upload_csv.pack()

listbox=tk.Listbox(classification_frame2)

# Pack the Listbox and button
listbox.pack()





# Switch to the main screen initially
go_back()


# Start the main event loop
root.mainloop()
