import mediapipe as mp
import cv2
import numpy as np
import time
from collections import deque
import tkinter as tk
from tkinter import messagebox, filedialog, Button
from PIL import Image, ImageTk, ImageSequence
import os
import csv
import sqlite3
from datetime import datetime

# Connect to the SQLite database
conn = sqlite3.connect('user_history.db')
c = conn.cursor()

# Create a table to store user history if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS user_history
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, login_time TEXT)''')

# Function to insert user history into the database
def insert_user_history(name):
    login_time = datetime.now().strftime('%d-%m-%Y %I:%M:%S %p') 
    c.execute("INSERT INTO user_history (name, login_time) VALUES (?, ?)", (name, login_time))
    conn.commit()

# Function to retrieve user history from the database
def get_user_history():
    c.execute("SELECT * FROM user_history")
    return c.fetchall()

# Function to detect the face
def DetectFace(root):
    # Load profile data
    reader = csv.DictReader(open('Profile.csv'))
    print('Detecting Login Face')
    for row in reader:
        result = dict(row)
        if result['ID'] == '1':
            name1 = result['Name']
        elif result['ID'] == '2':
            name2 = result['Name']

    # Load face recognition model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainData/Trainner.yml")

    # Load cascade classifier for face detection
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    
    # Initialize camera
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    Face_Id = ''

    # Function to preprocess an image
    def preprocess_image(image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        return equalized

    # Function to apply random rotation to an image
    def random_rotation(image):
        angle = np.random.randint(-15, 15)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))

    # Function to apply random scaling to an image
    def random_scaling(image):
        scale_factor = np.random.uniform(0.8, 1.2)
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)


    # Camera loop
    while True:
        ret, frame = cam.read()
        gray = preprocess_image(frame)  # Preprocess the captured frame
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Face detection loop
        if len(faces) > 0:  # Check if any face is detected
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                Face_Id = 'Not detected'
                Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                
                # Recognized face
                if confidence < 80:
                    if Id == 1:
                        Predicted_name = name1
                    elif Id == 2:
                        Predicted_name = name2
                    Face_Id = Predicted_name
                else:
                    Predicted_name = 'Unknown'
                    Face_Id = Predicted_name
                    
                cv2.putText(frame, str(Predicted_name), (x, y + h), font, 1, (255, 255, 255), 2)

            cv2.imshow('Picture', frame)
            cv2.waitKey(1)

            # Checking if the face matches for Login
            if Face_Id == 'Not detected':
                print("Face Not Detected. Try again.")
                pass
            elif Face_Id == name1 or Face_Id == name2 and Face_Id != 'Unknown':
                messagebox.showinfo("Welcome", f"Welcome, {Face_Id}.")
                print('Detected as {}. Login successful.'.format(Face_Id))
                print('Welcome, {}.'.format(Face_Id))
                insert_user_history(Face_Id)
                cam.release()
                cv2.destroyAllWindows()
                return True
            else:
                messagebox.showerror("Access Denied", "Unauthorized user. Access denied.")
                print('Login failed. Please try again.')
        
        else:
            messagebox.showerror("Access Denied", "No face detected. Try again.")
            print("No face detected. Try again.")
        
    cam.release()
    cv2.destroyAllWindows()
    return False

def paint_karo(root):
   
    # Constants
    ml = 150
    max_x, max_y = 250 + ml, 50
    curr_tool = "Choose Tool"
    time_init = True
    rad = 40
    var_inits = False
    thick = 4
    prevx, prevy = 0, 0
    drawing_color = (0, 0, 255)  # Default color is red
    max_drawings = 2000  # Maximum number of drawings to display in the "paint app" window

    
    # Get tools function
    def getTool(x):
        if x < 50 + ml:
            return "Line"
        elif x < 100 + ml:
            return "Rectangle"
        elif x < 150 + ml:
            return "Draw"
        elif x < 200 + ml:
            return "Circle"
        else:
            return "Eraser"

    def index_raised(i):
        yi = int(i.landmark[12].y * 480)
        y9 = int(i.landmark[9].y * 480)
        if (y9 - yi) > 40:
            return True
        return False

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
    draw = mp.solutions.drawing_utils

    # Drawing tools
    tools_path = "C:\Users\Deepak kumar\OneDrive\Desktop\Air canvas with face recognition\Images & gif\image\tools.png"
    tools = cv2.imread(tools_path)
    tools = tools.astype('uint8')

    # Create a white canvas
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

    cap = cv2.VideoCapture(0)

    # Constants for color palette
    color_palette = {
        (10, 80): (0, 0, 255),    # Red
        (10, 150): (0, 255, 0),    # Green
        (10, 220): (255, 0, 0),   # Blue
        (10, 290): (0, 255, 255)  # Yellow
    }

    # Position of the clear and save buttons
    color_button_height = 50
    button_spacing = 10  # Define the spacing between buttons
    clear_button_pos = (10, 10)  # Adjusted position for top-left corner
    save_button_pos = (10,310 + color_button_height + button_spacing)  

    # Add a variable to store the active color
    active_color = (0, 0, 255)  # Default color is red

    # Modify the update_drawing_color function to update the active color based on virtual touch
    def update_drawing_color(color):
        nonlocal drawing_color, active_color
        drawing_color = color
        active_color = color

    def is_within_button(x, y, button_pos):
        bx, by = button_pos
        return bx < x < bx + 50 and by < y < by + color_button_height

    def clear_canvas():
        nonlocal canvas
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

    def clear_all():
        nonlocal curr_tool, time_init, rad, var_inits, prevx, prevy, drawing_history
        curr_tool = "Choose Tool"
        time_init = True
        rad = 40
        var_inits = False
        prevx, prevy = 0, 0
        clear_canvas()
        drawing_history.clear()

    # Create a deque to hold drawings in the "paint app" window
    drawing_history = deque(maxlen=max_drawings)

    def draw_circle(img, center, radius, color, thickness):
        cv2.circle(img, center, radius, color, thickness)

    def save_drawing():
        nonlocal canvas
        try:
            # Open a file dialog to choose the save location and file format
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", ".jpg"), ("PNG files", ".png")])
            if save_path:
                # If the user selected a path, save the image
                cv2.imwrite(save_path, canvas)
                messagebox.showinfo("Success", "Drawing saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save drawing: {str(e)}")
    # Store the background video
    background_video = cv2.VideoCapture(0).read()[1]
    info_text = ""

    # Assuming DetectFace() is defined somewhere
    while True:
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)

        # Draw the current tool label on the right side of the toolbox
        tool_labels = {
            "Choose Tool": (300 + ml, 30),
            "Line": (300 + ml, 30),
            "Rectangle": (300 + ml, 30),
            "Draw": (300 + ml, 30),
            "Circle": (300 + ml, 30),
            "Eraser": (300 + ml, 30)
        }
        cv2.putText(frm, f"{curr_tool}", tool_labels[curr_tool], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        op = hand_landmark.process(rgb)

        if op.multi_hand_landmarks:
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
                x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

                if is_within_button(x, y, clear_button_pos):
                    clear_all()

                if is_within_button(x, y, save_button_pos):
                    save_drawing()

                # Check if hand is within any color in the palette
                for pos, color in color_palette.items():
                    if pos[0] < x < pos[0] + 50 and pos[1] < y < pos[1] + color_button_height:
                        update_drawing_color(color)
                        break
                
                if x < max_x and y < max_y and x > ml:
                    if time_init:
                        ctime = time.time()
                        time_init = False
                    ptime = time.time()

                    cv2.circle(frm, (x, y), rad, drawing_color, 2)
                    rad -= 1

                    if (ptime - ctime) > 0.8:
                        curr_tool = getTool(x)
                        print("Your current tool is set to:", curr_tool)
                        time_init = True
                        rad = 40
                else:
                    time_init = True
                    rad = 40

                if curr_tool == "Draw":
                    xi, yi = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

                    if index_raised(i):
                        if prevx !=0 and prevy !=0:
                            cv2.line(frm, (prevx, prevy), (xi, yi), drawing_color, thick)
                            cv2.line(canvas, (prevx, prevy), (xi, yi), drawing_color, thick)
                            drawing_history.append(("Draw", (prevx, prevy), (xi, yi), drawing_color, thick))
                        prevx, prevy = xi, yi
                    else:
                        cv2.circle(frm, (prevx, prevy), 2,drawing_color, -1)
                        cv2.circle(canvas, (prevx, prevy), 2,drawing_color, -1)
                        prevx,prevy =0,0

                elif curr_tool == "Line":
                    if index_raised(i):
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True

                        cv2.line(frm, (xii, yii), (x, y), drawing_color, thick)
                        cv2.line(canvas, (xii, yii), (x, y), drawing_color, thick)
                    else:
                        if var_inits:
                            var_inits = False
                            drawing_history.append(("Line", (xii, yii), (x, y), drawing_color, thick))

                elif curr_tool == "Rectangle":
                    if index_raised(i):
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True

                        cv2.rectangle(frm, (xii, yii), (x, y), drawing_color, thick)
                        cv2.rectangle(canvas, (xii, yii), (x, y), drawing_color, thick)
                    else:
                        if var_inits:
                            var_inits = False
                            drawing_history.append(("Rectangle", (xii, yii), (x, y), drawing_color, thick))

                elif curr_tool == "Circle":
                    if index_raised(i):
                        if not var_inits:
                            xii, yii = x, y
                            var_inits = True

                        radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                        draw_circle(frm, (xii, yii), radius, drawing_color, thick)
                        draw_circle(canvas, (xii, yii), radius, drawing_color, thick)
                    else:
                        if var_inits:
                            # Save the drawn circle to drawing history
                            drawing_history.append(("Circle", (xii, yii), radius, drawing_color, thick))
                            var_inits = False

                if curr_tool == "Eraser":
                    xi, yi = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

                    if index_raised(i):
                        for j in range(len(drawing_history) - 1, -1, -1):
                            if (drawing_history[j][1][0] - xi) ** 2 + (drawing_history[j][1][1] - yi) ** 2 < rad ** 2:
                                del drawing_history[j]
                        prevx, prevy = xi, yi
                    else:
                        prevx, prevy = 0, 0
                    # Don't draw on the frm (video feed) when using the eraser tool
                    frm[:max_y, ml:max_x] = background_video[:max_y, ml:max_x]                    
        # Draw the clear and save buttons
        cv2.rectangle(frm, clear_button_pos, (clear_button_pos[0] + 50, clear_button_pos[1] + color_button_height),
                      (255, 255, 255), -1)
        cv2.putText(frm, 'C', (clear_button_pos[0] + 25, clear_button_pos[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.rectangle(frm, save_button_pos, (save_button_pos[0] + 50, save_button_pos[1] + color_button_height),
                      (255, 255, 255), -1)
        cv2.putText(frm, 'S', (save_button_pos[0] + 25, save_button_pos[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display color palette in the "paint app" window
        for pos, color in color_palette.items():
            cv2.rectangle(frm, pos, (pos[0] + 50, pos[1] + color_button_height), color, -1)

        # Display toolbox
        frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

        # Clear the canvas before drawing
        canvas[:] = (255, 255, 255)

        # After the main loop, draw the drawing history on the "paint app" window
        for entry in drawing_history:
            if entry[0] == "Draw":
                cv2.line(frm, entry[1], entry[2], entry[3], entry[4])
                cv2.line(canvas, entry[1], entry[2], entry[3], entry[4])
            elif entry[0] == "Line":
                cv2.line(frm, entry[1], entry[2], entry[3], entry[4])
                cv2.line(canvas, entry[1], entry[2], entry[3], entry[4])
            elif entry[0] == "Rectangle":
                cv2.rectangle(frm, entry[1], entry[2], entry[3], entry[4])
                cv2.rectangle(canvas, entry[1], entry[2], entry[3], entry[4])
            elif entry[0] == "Circle":
                draw_circle(frm, entry[1], entry[2], entry[3], entry[4])
                draw_circle(canvas, entry[1], entry[2], entry[3], entry[4])
            elif entry[0] == "Eraser":
                if isinstance(entry[2], int):
                    cv2.circle(frm, entry[1], entry[2], (255, 255, 255), -1)
                    cv2.circle(canvas, entry[1], entry[2], (255, 255, 255), -1)
                else:
                    cv2.line(frm, entry[1], entry[2], (255, 255, 255), thickness=thick)
                    cv2.line(canvas, entry[1], entry[2], (255, 255, 255), thickness=thick)


        # Display canvas in the "paint app" window
        cv2.imshow("paint app", frm)

        # Display canvas in the "canvas" window
        cv2.imshow("canvas", canvas)

        background_video = cap.read()[1]


        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
def open_history_window():
    # Create a new top-level window for history
    history_window = tk.Toplevel()
    history_window.title("History")
    history_window.configure(bg="black")
    history_window.geometry("2000x1000")

    # Add content to the history window (you can customize this according to your needs)
    history_label = tk.Label(history_window, text="⏪  Echoes of your journey  ⏪", bg="black", fg="green", font=("Showcard Gothic", 25,"bold"))
    history_label.pack(pady=20)

    hi = r"C:\Users\Deepak kumar\OneDrive\Desktop\Air canvas with face recognition\Images & gif\gif\3HeZ.gif"
    gif = Image.open(hi)

    frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(gif)]
    
    # Place the GIF in the top right corner
    gif_x = history_window.winfo_screenwidth() - gif.width
    gif_y = 0

    # Keep references to the PhotoImage objects
    history_window.image_refs = frames

    gif_label = tk.Label(history_window, bg="black")
    gif_label.image = frames[0]  # Keep a reference to the PhotoImage object
    gif_label.config(image=frames[0])
    gif_label.place(x=gif_x, y=gif_y)

    def update_gif(frame_index):
        # Display the frame
        gif_label.config(image=frames[frame_index])
        # Increment the frame index
        frame_index += 1
        # Loop back to the first frame if at the end
        if frame_index == len(frames):
            frame_index = 0
        # Schedule the next update after 100 milliseconds
        history_window.after(100, update_gif, frame_index)

    # Start the GIF animation
    update_gif(0)
    user_history = get_user_history()

    # Display user history in the window
    for user in user_history:
        user_info = f"Name: {user[1]}, Login Time: {user[2]}"
        history_entry = tk.Label(history_window, text=user_info, bg="black", fg="white", font=("Arial", 12))
        history_entry.pack(pady=5)
     # Function to clear history data from the database
    def clear_history():
        if messagebox.askyesno("Clear History", "Are you sure you want to clear the history?"):
            # Execute SQL command to delete all records from the user_history table
            c.execute("DELETE FROM user_history")
            conn.commit()
            #for widget in history_window.winfo_children():
                #widget.destroy()
            # Show a message indicating that history has been cleared
            messagebox.showinfo("History Cleared", "User history has been cleared successfully.")

    # Add a "Clear" button at the bottom of the history window
    clear_button = Button(history_window, text="Clear", command=clear_history, bg="green", fg="black", font=("Showcard Gothic", 25,"bold"))
    clear_button.place(relx=0.42, rely=0.85)   
        


def start_process():
    root = tk.Tk()
    root.title("WELCOME BACK")
    root.configure(bg="black")
    # Set window size to fit the whole screen
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

    # Load the GIF image
    gif_path =r"C:\Users\Deepak kumar\OneDrive\Desktop\Air canvas with face recognition\Images & gif\gif\4Mg1 (1).gif"

    gif = Image.open(gif_path)

    # Create a list of frames from the GIF
    frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(gif)]

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    gif_x = (screen_width - gif.width) // 2
    gif_y = (screen_height - gif.height) // 2

    # Create a label to display the GIF
    gif_label = tk.Label(root, bg="black", image=frames[0])
    gif_label.place(x=gif_x, y=gif_y)

    # Function to update the GIF
    def update_gif(frame_index):
        # Display the frame
        gif_label.config(image=frames[frame_index])
        # Increment the frame index
        frame_index += 1
        # Loop back to the first frame if at the end
        if frame_index == len(frames):
            frame_index = 0
        # Schedule the next update after 100 milliseconds
        root.after(100, update_gif, frame_index)

    # Start the GIF animation
    update_gif(0)

    # Create a title label
    title = tk.Label(root, text="YOUR DIGITAL WORKSPACE AWAITS", bg="red", fg="black", font=("times new roman", 25, "bold"), relief=tk.GROOVE, bd=12)
    title.pack(fill=tk.X)

    # Create a button with custom styling
    start_button = Button(root, text="Get Started", command=lambda: start_face_detection(root), font=("Arial", 14,"bold"), bg="red", fg="black",relief="sunken",bd=8, width=20, height=2)
    start_button.place(relx=0.20, rely=0.5)

    his_button = Button(root, text="HISTORY", font=("Arial", 14,"bold"), bg="red", fg="black",relief="sunken",bd=8, width=20, height=2 ,command=open_history_window)
    his_button.place(relx=0.65, rely=0.5)

    exit_button = Button(root, text="EXIT", font=("Arial", 14,"bold"), bg="red", fg="black",relief="sunken",bd=8, width=20, height=2, command=lambda: exit_program(root))
    exit_button.place(relx=0.42, rely=0.75)

    root.mainloop()# Run the Tkinter event loop
def exit_program(root):
    if tk.messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()


def start_face_detection(root):
    if DetectFace(root):
        root.destroy()
        paint_karo(root)
       
    else:
        messagebox.showerror("Access Denied", "Face not recognized. Access denied.")

# Start the process
start_process()
