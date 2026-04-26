import cv2
import os
import sys
import time

# ASCII gradient from Dark to Light (you can reverse this if you have a light terminal background)
ASCII_CHARS = ["@", "%", "#", "*", "+", "=", "-", ":", ".", " "]

def frame_to_ascii(frame, cols):
    """Converts a BGR OpenCV frame into a string of ASCII characters."""
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Calculate dimensions. Terminal characters are roughly twice as tall 
    # as they are wide, so we multiply by 0.5 to fix the aspect ratio.
    aspect_ratio = gray.shape[0] / gray.shape[1]
    rows = int(cols * aspect_ratio * 0.5)
    
    # 3. Resize image to fit the terminal
    resized = cv2.resize(gray, (cols, rows))
    
    # 4. Map pixels to ASCII
    ascii_frame = ""
    for row in resized:
        for pixel in row:
            # Map the 0-255 pixel brightness to our 0-9 ASCII_CHARS list
            ascii_frame += ASCII_CHARS[min(pixel // 26, 9)]
        ascii_frame += "\n"
        
    return ascii_frame

def main():
    print("Initializing physical camera...")
    
    # Try index 0 (standard USB webcams and older Pi setups)
    # Try index 0 (standard USB webcams and older Pi setups)
    cap = cv2.VideoCapture('libcamerasrc ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("🚨 Error: Could not open camera at index 0.")
        print("If you are using the official RPi Ribbon camera on a Pi 5,")
        print("you may need to change the code to:")
        print("cap = cv2.VideoCapture('libcamerasrc ! appsink')")
        return

    # Get the current size of the SSH terminal window
    try:
        columns, _ = os.get_terminal_size()
    except OSError:
        columns = 80 # Fallback if terminal size can't be read
        
    width = columns - 2 # Leave a tiny margin so it doesn't wrap weirdly

    print("Camera active! Press Ctrl+C to exit.")
    time.sleep(1) # Brief pause to read the message

    # Clear the terminal screen once before we start drawing
    os.system('clear')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab a frame from the camera.")
                break
                
            # Convert frame
            ascii_art = frame_to_ascii(frame, width)
            
            # Use ANSI escape code to move the cursor to the top-left (0,0) 
            # instead of clearing the screen. This prevents terrible screen flickering!
            sys.stdout.write('\033[H')
            sys.stdout.write(ascii_art)
            sys.stdout.flush()
            
            # Cap the framerate slightly so it doesn't overwhelm the SSH connection
            time.sleep(0.03) 
            
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        
    finally:
        cap.release()
        os.system('clear')
        print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()