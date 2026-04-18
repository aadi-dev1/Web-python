import cv2
import face_recognition
import numpy as np
import pickle
import os
import csv
from datetime import datetime


# Utility function to load face data
def load_encodings(data_file="face_data.pkl"):
    try:
        with open(data_file, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}


# Utility function to save face data
def save_encodings(data, data_file="face_data.pkl"):
    with open(data_file, "wb") as f:
        pickle.dump(data, f)


# 1. Capture Sample
def capture_samples(cap_sample=10, data_file="face_data.pkl", image_folder="face_images"):
    # Load existing face data
    face_data = load_encodings(data_file)
    cap = cv2.VideoCapture(0)
    encodings = []
    sample_count = 0

    # Get name and generate unique ID
    name = input("Enter person name: ")
    existing_ids = [key[3:] for key in face_data.keys()]
    new_id = f"KRV{len(existing_ids) + 1}"

    # Create folder if not exists
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    while sample_count < cap_sample:
        ret, frame = cap.read()
        cv2.imshow("Capturing Samples", frame)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)

        if len(face_locations) > 1:
            print("Multiple faces detected. Please ensure only one person is in the frame.")
            cv2.putText(frame, "Multiple faces detected! Ensure only one person is visible.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Capturing Samples", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Trigger sample capture with 'c' for the first sample
        if sample_count == 0 and cv2.waitKey(1) & 0xFF == ord('c'):
            face_encoding = face_recognition.face_encodings(frame, face_locations)

            if face_encoding:
                # Check for duplicate faces
                for known_encoding in face_data.values():
                    if face_recognition.compare_faces([known_encoding["encoding"]], face_encoding[0], tolerance=0.4)[0]:
                        print(f"Warning: Face already exists in the dataset as '{known_encoding['name']}'.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Save the first sample image
                image_path = os.path.join(image_folder, f"{new_id}_{name}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"First sample image saved at {image_path}.")

                encodings.append(face_encoding[0])
                sample_count += 1
                print(f"Sample {sample_count}/{cap_sample} captured.")

        # Automatically capture the remaining samples
        elif sample_count > 0:
            face_encoding = face_recognition.face_encodings(frame, face_locations)

            if face_encoding:
                encodings.append(face_encoding[0])
                sample_count += 1
                print(f"Sample {sample_count}/{cap_sample} captured.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save captured face data
    if encodings:
        average_encoding = np.mean(encodings, axis=0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        face_data[new_id] = {"name": name, "encoding": average_encoding, "timestamp": timestamp}
        save_encodings(face_data, data_file)
        print(f"Encodings for {name} (ID: {new_id}) saved successfully at {timestamp}.")

    cap.release()
    cv2.destroyAllWindows()

# 2. Real-Time Face Recognition
def run_face_recognition(data_file="face_data.pkl"):
    face_data = load_encodings(data_file)
    known_face_names = [f"{key} - {val['name']}" for key, val in face_data.items()]
    known_face_encodings = [val["encoding"] for val in face_data.values()]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        frame = draw_label(frame, face_locations, face_names)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 3. View Records
def view_records(data_file="face_data.pkl"):
    face_data = load_encodings(data_file)
    for id, info in face_data.items():
        print(f"ID: {id}, Name: {info['name']}, Timestamp: {info['timestamp']}")


# 4. Modify Records
def modify_records(data_file="face_data.pkl", image_folder="face_images"):
    # Load existing face data
    face_data = load_encodings(data_file)

    if not face_data:
        print("No records found to modify.")
        return

    # Display all IDs and names
    print("Available Records:")
    for id, info in face_data.items():
        print(f"ID: {id}, Name: {info['name']}")

    # Prompt for ID to modify
    id_to_modify = input("Enter ID to modify: ")

    if id_to_modify in face_data:
        current_name = face_data[id_to_modify]['name']
        image_filename = f"{id_to_modify}_{current_name}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        # Modify the name
        new_name = input(f"Enter new name for {current_name} (or press Enter to keep the same): ").strip()
        if new_name:
            face_data[id_to_modify]['name'] = new_name
            print(f"Name updated to '{new_name}'.")

        # Ask if the user wants to update the encodings
        update_encodings = input("Do you want to update face encodings? (y/n): ").strip().lower()
        if update_encodings == 'y':
            # Delete the existing image if it exists
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Existing image '{image_filename}' deleted.")

            # Capture new face encodings and save the updated image
            cap = cv2.VideoCapture(0)
            encodings = []
            sample_count = 0
            cap_sample = 10

            print("Starting sample collection. Press 'c' for the first sample.")
            while sample_count < cap_sample:
                ret, frame = cap.read()
                if not ret:
                    print("Error accessing camera.")
                    break

                cv2.imshow("Updating Encodings", frame)
                key = cv2.waitKey(1)

                # For the first sample, save the image
                if sample_count == 0 and key & 0xFF == ord('c'):
                    new_image_filename = f"{id_to_modify}_{face_data[id_to_modify]['name']}.jpg"
                    new_image_path = os.path.join(image_folder, new_image_filename)
                    cv2.imwrite(new_image_path, frame)
                    print(f"New image saved as '{new_image_filename}'.")

                # Process all samples for encodings
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    sample_count += 1
                    print(f"Sample {sample_count}/{cap_sample} captured.")

                if sample_count >= cap_sample:
                    break

            cap.release()
            cv2.destroyAllWindows()

            # Update encoding if samples were successfully captured
            if encodings:
                average_encoding = np.mean(encodings, axis=0)
                face_data[id_to_modify]['encoding'] = average_encoding
                print("Encodings updated successfully.")
            else:
                print("No encodings captured. Existing encodings remain unchanged.")
        else:
            print("Encodings update skipped.")

        # Save the updated record
        save_encodings(face_data, data_file)
        print(f"Record for ID {id_to_modify} updated successfully.")
    else:
        print("ID not found.")


# 5. Delete Records
def delete_record(data_file="face_data.pkl", image_folder="face_images"):
    # Load existing face data
    face_data = load_encodings(data_file)

    if not face_data:
        print("No records found to delete.")
        return

    # Display all IDs and names
    print("Available Records:")
    for id, info in face_data.items():
        print(f"ID: {id}, Name: {info['name']}")

    # Prompt for ID to delete
    id_to_delete = input("Enter ID to delete: ")

    if id_to_delete in face_data:
        # Construct the image file name using the format "ID_Name.jpg"
        name = face_data[id_to_delete]['name']
        image_filename = f"{id_to_delete}_{name}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        # Show the image for confirmation (if it exists)
        if os.path.exists(image_path):
            print(f"Displaying image for ID: {id_to_delete}")
            img = cv2.imread(image_path)
            cv2.imshow(f"Image for {id_to_delete} - {name}", img)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"No image found for ID {id_to_delete} with the name '{name}' in {image_folder}.")

        # Ask for confirmation
        confirm = input(f"Are you sure you want to delete the record and image for {name} (ID: {id_to_delete})? (y/n): ").strip().lower()
        if confirm == 'y':
            # Delete the image if it exists
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Image '{image_filename}' deleted from {image_folder}.")
            
            # Delete the record
            del face_data[id_to_delete]
            save_encodings(face_data, data_file)
            print(f"Record for ID {id_to_delete} deleted successfully.")
        else:
            print("Deletion cancelled.")
    else:
        print("ID not found.")




# 6. Attendance Module
def log_attendance(name, log_file="attendance_log.csv"):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    print(f"Attendance logged for {name}.")


# 7. View Attendance Log
def view_attendance_log(attendance_file="attendance.pkl"):
    try:
        with open(attendance_file, "rb") as f:
            attendance_data = pickle.load(f)

        if not attendance_data:
            print("No attendance records found.")
            return

        print(f"{'ID':<10}{'Name':<25}{'Timestamp':<25}{'Status':<10}")
        print("-" * 70)
        for person_id, details in attendance_data.items():
            name = details.get("name", "Unknown")
            timestamp = details.get("timestamp", "N/A")
            status = details.get("status", "A")  # Default to absent if no status
            print(f"{person_id:<10}{name:<25}{timestamp:<25}{status:<10}")

    except FileNotFoundError:
        print("Attendance file not found. No records to display.")
    except EOFError:
        print("Attendance file is empty. No records to display.")



# 8. Review Unrecognized Faces
def review_unrecognized_faces(folder="unrecognized_faces"):
    if not os.path.exists(folder):
        print("No unrecognized faces found.")
        return

    for file in os.listdir(folder):
        print(f"Reviewing: {file}")
        img = cv2.imread(os.path.join(folder, file))
        cv2.imshow("Unrecognized Face", img)
        if cv2.waitKey(0) & 0xFF == ord('d'):
            os.remove(os.path.join(folder, file))
            print(f"{file} deleted.")
        cv2.destroyAllWindows()
        
        
def draw_label(frame, face_locations, face_names):
    """
    Draw rectangles around faces and display bold labels in boxes on the video frame.

    :param frame: The video frame from the camera.
    :param face_locations: A list of face locations detected in the frame.
    :param face_names: A list of names corresponding to the detected faces.
    :return: The updated video frame with labels.
    """
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Get text size for label box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1  # Set a higher thickness for bold text
        text_size, _ = cv2.getTextSize(name, font, font_scale, font_thickness)

        # Define label box dimensions
        label_height = text_size[1] + 8
        label_width = text_size[0] + 8

        # Draw label box (filled rectangle)
        cv2.rectangle(frame, (left, bottom), (left + label_width, bottom + label_height), (0, 255, 0), cv2.FILLED)

        # Add bold text to label box (drawn only once)
        text_x = left + 5
        text_y = bottom + label_height - 5
        cv2.putText(frame, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return frame


def mark_attendance(data_file="face_data.pkl", attendance_file="attendance.pkl", unknown_data_file="unknown_face_data.pkl", unknown_folder="unknown_faces"):
    # Load face data
    face_data = load_encodings(data_file)
    unknown_face_data = load_encodings(unknown_data_file)

    # Initialize attendance dictionary if not already present
    try:
        with open(attendance_file, "rb") as f:
            attendance_data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        attendance_data = {}

    # Prepare video capture
    cap = cv2.VideoCapture(0)
    known_face_encodings = [val["encoding"] for val in face_data.values()]
    known_face_ids = [key for key in face_data.keys()]
    known_face_names = [val["name"] for val in face_data.values()]

    print("Press 'q' to exit the attendance module.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing camera.")
            break

        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if face matches a known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            id = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                # Known face
                id = known_face_ids[best_match_index]
                name = known_face_names[best_match_index]

                # Check last detection time
                current_time = datetime.now()
                if id in attendance_data:
                    last_detection = attendance_data[id].get("last_detected")
                    time_diff = (current_time - datetime.strptime(last_detection, "%Y-%m-%d %H:%M:%S")).total_seconds()
                    if time_diff <= 1800:  # 30 minutes in seconds
                        label = f"Attendance already marked: {id} - {name}"
                    else:
                        attendance_data[id]["timestamp"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        attendance_data[id]["status"] = "P"
                        label = f"Attendance Marked: {id} - {name}"
                else:
                    # First-time detection during this session
                    attendance_data[id] = {
                        "name": name,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "P",
                        "last_detected": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    label = f"Attendance Marked: {id} - {name}"
            else:
                # Handle unknown face
                unknown_id = None
                for key, value in unknown_face_data.items():
                    if face_recognition.compare_faces([value["encoding"]], face_encoding, tolerance=0.5)[0]:
                        unknown_id = key
                        break

                if not unknown_id:
                    unknown_id = f"KRVU{len(unknown_face_data) + 1}"
                    unknown_face_data[unknown_id] = {"encoding": face_encoding}
                    os.makedirs(unknown_folder, exist_ok=True)
                    unknown_image_path = os.path.join(unknown_folder, f"{unknown_id}.jpg")
                    top, right, bottom, left = face_location
                    cv2.imwrite(unknown_image_path, frame[top:bottom, left:right])
                    label = f"Unknown Face Detected: {unknown_id}"
                    print(f"Unknown face saved: {unknown_id} at {unknown_image_path}")
                else:
                    label = f"Unknown Face Already Detected: {unknown_id}"
                    print(f"Unknown face already recorded as {unknown_id}.")

            face_names.append(label)
            if id != "Unknown":
                attendance_data[id]["last_detected"] = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Display frame with face labels
        frame = draw_label(frame, face_locations, face_names)
        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save updated attendance and unknown face data
    with open(attendance_file, "wb") as f:
        pickle.dump(attendance_data, f)
    save_encodings(unknown_face_data, unknown_data_file)
    print("Attendance process completed and saved.")



# Main Menu
while True:

    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("""
        .-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-.
        .~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.
        !_.                                             J.P. EDUCATION ACADEMY                                          ._!
        .~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.
        >_<.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-   K R U T R I M _._ V I S I O N   .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.->_<
        !                                        FACE RECOGNITION  & ATTENDANCE SYSTEM                                    !   
        >.<------------------------------------------------------------------------------------------------------------->.<
        *                                                                                                                 *
        *                           1. Capture Sample                               5. Real-Time Face Recognition         *
        *                           2. View Records                                 6. Run Attendance Module              *
        *                           3. Modify Records                               7. View Attendance Log                *
        *                           4. Delete Records                               8. Review Unrecognized Faces          *
        *                                                       0. Exit                                                   * 
        *-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-*
        """)
    choice = int(input("\t\t\t\t\tEnter choice: "))
    if choice == 1:
        capture_samples()
        input("\n\tPress Enter Key...")
    elif choice == 2:
        view_records()
        input("\n\tPress Enter Key...")
    elif choice == 3:
        modify_records()
        input("\n\tPress Enter Key...")
    elif choice == 4:
        delete_record()
        input("\n\tPress Enter Key...")
    elif choice == 5:
        run_face_recognition()
    elif choice == 6:
        mark_attendance()
        input("\n\tPress Enter Key...")
    elif choice == 7:
        view_attendance_log()
        input("\n\tPress Enter Key...")
    elif choice == 8:
        review_unrecognized_faces()
        input("\n\tPress Enter Key...")
    elif choice == 0:
        print("Thank you! Bye!")
        break
    else:
        print("Invalid choice. Try again.")

