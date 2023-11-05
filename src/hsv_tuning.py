import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


def copy_values_to_clipboard():
    hsv_text = ""
    ycrcb_text = ""
    if use_hsv_var.get():
        hsv_text = f"(np.array({hsv_lower}), np.array({hsv_upper}))"
        root.clipboard_clear()
        root.clipboard_append(hsv_text)
    if use_ycrcb_var.get():
        ycrcb_text = f"(np.array({ycrcb_lower}), np.array({ycrcb_upper}))"
        root.clipboard_clear()
        root.clipboard_append(ycrcb_text)
    root.update()


def update_values_label():
    hsv_text = ""
    ycrcb_text = ""
    if use_hsv_var.get():
        hsv_text = f"HSV: (np.array({hsv_lower}), np.array({hsv_upper}))"
    if use_ycrcb_var.get():
        ycrcb_text = f"YCrCb: (np.array({ycrcb_lower}), np.array({ycrcb_upper}))"
    values_label.config(text=f"{hsv_text}\n{ycrcb_text}")

def update_hsv_range(event=None):
    global hsv_lower, hsv_upper, image, hsv_image

    if use_hsv_var.get():
        # Get the lower and upper HSV values from the sliders
        hue_low = hue_low_slider.get()
        hue_high = hue_high_slider.get()
        sat_low = sat_low_slider.get()
        sat_high = sat_high_slider.get()
        val_low = val_low_slider.get()
        val_high = val_high_slider.get()

        # Update the HSV range
        hsv_lower = (hue_low, sat_low, val_low)
        hsv_upper = (hue_high, sat_high, val_high)

        # Create a mask based on the HSV range
        mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        display_result(result)
        update_values_label()

    else:
        display_result(image)


def update_ycrcb_range(event=None):
    global ycrcb_lower, ycrcb_upper, image, ycrcb_image

    if use_ycrcb_var.get():
        # Get the lower and upper YCrCb values from the sliders
        y_low = y_low_slider.get()
        y_high = y_high_slider.get()
        cr_low = cr_low_slider.get()
        cr_high = cr_high_slider.get()
        cb_low = cb_low_slider.get()
        cb_high = cb_high_slider.get()

        # Update the YCrCb range
        ycrcb_lower = (y_low, cr_low, cb_low)
        ycrcb_upper = (y_high, cr_high, cb_high)

        # Create a mask based on the YCrCb range
        mask = cv2.inRange(ycrcb_image, ycrcb_lower, ycrcb_upper)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        display_result(result)
        update_values_label()

    else:
        display_result(image)


def display_result(result):
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)
    result_tk = ImageTk.PhotoImage(result_pil)
    image_label.config(image=result_tk)
    image_label.image = result_tk


# Create the main window
root = tk.Tk()
root.title("Color Space Range Tracker")

# Load an image
image_path = filedialog.askopenfilename()
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Initialize HSV and YCrCb range values
hsv_lower = (0, 0, 0)
hsv_upper = (179, 255, 255)
ycrcb_lower = (0, 0, 0)
ycrcb_upper = (255, 255, 255)

# Create frames for each color space
hsv_frame = ttk.Frame(root)
hsv_frame.pack(side=tk.LEFT, padx=10, pady=10)

ycrcb_frame = ttk.Frame(root)
ycrcb_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Add checkboxes to toggle the filters
use_hsv_var = tk.IntVar()
use_hsv_checkbox = ttk.Checkbutton(hsv_frame, text="Use HSV Filter", variable=use_hsv_var, command=update_hsv_range)
use_hsv_checkbox.pack()

use_ycrcb_var = tk.IntVar()
use_ycrcb_checkbox = ttk.Checkbutton(ycrcb_frame, text="Use YCrCb Filter", variable=use_ycrcb_var, command=update_ycrcb_range)
use_ycrcb_checkbox.pack()

# Create sliders for HSV range and pack them into hsv_frame
hue_low_slider = tk.Scale(hsv_frame, from_=0, to=179, label="Hue Low", orient="horizontal", length=300, command=update_hsv_range)
hue_high_slider = tk.Scale(hsv_frame, from_=0, to=179, label="Hue High", orient="horizontal", length=300, command=update_hsv_range)
sat_low_slider = tk.Scale(hsv_frame, from_=0, to=255, label="Sat Low", orient="horizontal", length=300, command=update_hsv_range)
sat_high_slider = tk.Scale(hsv_frame, from_=0, to=255, label="Sat High", orient="horizontal", length=300, command=update_hsv_range)
val_low_slider = tk.Scale(hsv_frame, from_=0, to=255, label="Val Low", orient="horizontal", length=300, command=update_hsv_range)
val_high_slider = tk.Scale(hsv_frame, from_=0, to=255, label="Val High", orient="horizontal", length=300, command=update_hsv_range)

hue_low_slider.set(hsv_lower[0])
hue_high_slider.set(hsv_upper[0])
sat_low_slider.set(hsv_lower[1])
sat_high_slider.set(hsv_upper[1])
val_low_slider.set(hsv_lower[2])
val_high_slider.set(hsv_upper[2])


# Display the HSV sliders
hue_low_slider.pack()
hue_high_slider.pack()
sat_low_slider.pack()
sat_high_slider.pack()
val_low_slider.pack()
val_high_slider.pack()

# Create sliders for YCrCb range and pack them into ycrcb_frame
y_low_slider = tk.Scale(ycrcb_frame, from_=0, to=255, label="Y Low", orient="horizontal", length=300, command=update_ycrcb_range)
y_high_slider = tk.Scale(ycrcb_frame, from_=0, to=255, label="Y High", orient="horizontal", length=300, command=update_ycrcb_range)
cr_low_slider = tk.Scale(ycrcb_frame, from_=0, to=255, label="Cr Low", orient="horizontal", length=300, command=update_ycrcb_range)
cr_high_slider = tk.Scale(ycrcb_frame, from_=0, to=255, label="Cr High", orient="horizontal", length=300, command=update_ycrcb_range)
cb_low_slider = tk.Scale(ycrcb_frame, from_=0, to=255, label="Cb Low", orient="horizontal", length=300, command=update_ycrcb_range)
cb_high_slider = tk.Scale(ycrcb_frame, from_=0, to=255, label="Cb High", orient="horizontal", length=300, command=update_ycrcb_range)

y_low_slider.set(ycrcb_lower[0])
y_high_slider.set(ycrcb_upper[0])
cr_low_slider.set(ycrcb_lower[1])
cr_high_slider.set(ycrcb_upper[1])
cb_low_slider.set(ycrcb_lower[2])
cb_high_slider.set(ycrcb_upper[2])

# Display the YCrCb sliders
y_low_slider.pack()
y_high_slider.pack()
cr_low_slider.pack()
cr_high_slider.pack()
cb_low_slider.pack()
cb_high_slider.pack()

# Create a label to display the image
image_label = ttk.Label(root)
image_label.pack()

# Add a label to display the current HSV and YCrCb values
values_label = ttk.Label(root, text="", wraplength=400)
values_label.pack(pady=20)

# Add a button to copy the values to clipboard
copy_button = ttk.Button(root, text="Copy Values", command=copy_values_to_clipboard)
copy_button.pack(pady=10)

# Initialize the display with the original image
display_result(image)

# Run the main loop
root.mainloop()
