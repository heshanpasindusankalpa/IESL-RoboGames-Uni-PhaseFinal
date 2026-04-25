import cv2

def run_viewer():
    print("Connecting to the drone's live stream...")
    
    # The address where flight.py is broadcasting
    stream_url = "http://192.168.43.171:5000" 
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not connect. Make sure flight.py is running on the Pi!")
        return

    print("Stream connected! Press 'q' on your keyboard to close the window.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream disconnected or lost.")
            break

        # Display the frame in a standalone window
        cv2.imshow("Drone HUD Live Stream", frame)

        # Check for the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_viewer()