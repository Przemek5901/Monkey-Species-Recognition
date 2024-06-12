import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module='keras')

class TestSinglePicture:
    def __init__(self, master=None):
        # Load the model
        self.model = load_model('monkey_species_recognition_model.keras')
        self.class_mapping = {
            0: 'mantled howler',
            1: 'patas monkey',
            2: 'bald uakari',
            3: 'japanese macaque',
            4: 'pygmy marmoset',
            5: 'white headed capuchin',
            6: 'silvery marmoset',
            7: 'common squirrel monkey',
            8: 'black headed night monkey',
            9: 'nilgiri langur',
        }

        # Build UI
        self.callback = None
        toplevel1 = tk.Tk() if master is None else tk.Toplevel(master)
        toplevel1.title("Monkey Species Recognition")
        toplevel1.iconbitmap('D:\\Monkey-Species-Recognition\\Monkey Species Recognition\\icon.ico')
        toplevel1.configure(height=200, width=200)
        toplevel1.geometry("650x480")
        self.browseButton = ttk.Button(toplevel1, name="browsebutton")
        self.browseButton.configure(
            compound="left",
            cursor="arrow",
            default="normal",
            state="normal",
            takefocus=False,
            text='Przeglądaj',
            width=30)
        self.browseButton.place(anchor="nw", x=230, y=380)
        self.browseButton.bind("<Activate>", self.callback, add="")
        message1 = tk.Message(toplevel1)
        message1.configure(
            cursor="based_arrow_down",
            font="{Arial} 14 {}",
            relief="flat",
            text='Wybierz zdjęcie do rozpoznania\n',
            width=500)
        message1.place(anchor="nw", x=180, y=10)

        toplevel1.grid_anchor("n")
        toplevel1.rowconfigure(0, pad=2)

        self.image_label = tk.Label(toplevel1)
        self.image_label.place(anchor="nw", x=170, y=55)

        self.result_label = tk.Label(toplevel1, font="{Arial Baltic} 14 {}")
        self.result_label.place(anchor="nw", x=215, y=410)

        self.browseButton.configure(command=self.browse_files)

        # Main widget
        self.mainwindow = toplevel1

    def browse_files(self):
        # Open file dialog
        file_name = filedialog.askopenfilename(title="Wybierz plik", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_name:
            image = Image.open(file_name)
            image = image.resize((300, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.browseButton.configure(text="Przeglądaj ponownie")


            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Make prediction
            predicted_brand = self.predict_image(file_name)
            self.result_label.config(text=f"Rozpoznany gatunek:\n {predicted_brand}")

    def predict_image(self, img_path):
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        return self.class_mapping.get(predicted_class, 'Unknown')

    def run(self):
        self.mainwindow.mainloop()

if __name__ == "__main__":
    app = TestSinglePicture()
    app.run()
