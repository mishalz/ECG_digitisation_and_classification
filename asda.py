
# root = tk.Tk()
# root.title("Upload ECG Image")
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

# # Set the width and height of the window to the screen size
# root.geometry("%dx%d" % (screen_width, screen_height))
# # style = ThemedStyle(root)
# # style.theme_use('arc')

# heading = tk.Label(root, text="ECG Digitization And Classification", font=(
#     "Arial", 20, "bold"), bg="#ADCFCD")
# heading.place(relx=0.49, rely=0.3, anchor=tk.CENTER)


# # Create a photoimage object of the image in the path
# photo = ImageTk.PhotoImage(Image.open(r"digitization.png").resize((100, 100)))
# image_label = tk.Label(image=photo)
# digitization_button = tk.Button(root, image=photo, command=digitization_window).place(
#     relx=0.41, rely=0.47, anchor=tk.CENTER)

# dig_label = tk.Label(root, text='Digitize ECG',
#                      font=("Arial", 12, 'bold'), bg="#ADCFCD")
# dig_label.place(relx=0.41, rely=0.59, anchor=tk.CENTER)

# photo1 = ImageTk.PhotoImage(Image.open(
#     r"classification.png").resize((100, 100)))
# image_label1 = tk.Label(image=photo1)
# classification_button = tk.Button(root, image=photo1, command=classification_window).place(
#     relx=0.57, rely=0.47, anchor=tk.CENTER)

# classify_label = tk.Label(root, text="Classify ECG",
#                           font=("Arial", 12, 'bold'), bg="#ADCFCD")
# classify_label.place(relx=0.57, rely=0.59, anchor=tk.CENTER)


# root.configure(bg="#ADCFCD")
# root.mainloop()
