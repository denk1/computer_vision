import face_recognition
import cv2
from encoding_face import EncodingFace
import pickle

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

# denis_image = face_recognition.load_image_file("denis.jpg")
# denis_face_encoding = face_recognition.face_encodings(denis_image)[0]
# sasha_image = face_recognition.load_image_file("sasha.jpg")
# sasha_face_encoding = face_recognition.face_encodings(sasha_image)[0]
# anatolio_image = face_recognition.load_image_file("anatolio.jpg")
# anatolio_face_encoding = face_recognition.face_encodings(anatolio_image)[0]
# dveselov_image = face_recognition.load_image_file("dima_veselov.jpg")
# dveselov_face_encoding = face_recognition.face_encodings(dveselov_image)[0]
# dgrishin_image = face_recognition.load_image_file("dima_grishin.jpg")
# dgrishin_face_encoding = face_recognition.face_encodings(dgrishin_image)[0]
# leha_image = face_recognition.load_image_file("leha.jpg")
# leha_face_encoding = face_recognition.face_encodings(leha_image)[0]
# kirill_image = face_recognition.load_image_file("kirill.jpg")
# kirill_face_encoding = face_recognition.face_encodings(kirill_image)[0]
# soloviev_image = face_recognition.load_image_file("sasha_soloviev.jpg")
# soloviev_face_encoding = face_recognition.face_encodings(soloviev_image)[0]
# gurgen_image = face_recognition.load_image_file("gurgen.jpg")
# gurgen_face_encoding = face_recognition.face_encodings(gurgen_image)[0]
# andrey_image = face_recognition.load_image_file("andrey.jpg")
# andrey_face_encoding = face_recognition.face_encodings(andrey_image)[0]

list_thread_recognise = [EncodingFace('denis.jpg', 'Denis Kozmin'),\
                         EncodingFace('sasha.jpg', 'Alexander Ryabin'),\
                         EncodingFace('anatolio.jpg', 'Anatoliy Mihaylovich'),\
                         EncodingFace('dima_veselov.jpg', 'Dmitriy Veselov'),\
                         EncodingFace('leha.jpg', 'Aleksey Svechkar'),\
                         EncodingFace('sasha_soloviev.jpg', 'Aleksandr Soloviev'),\
                         EncodingFace('gurgen.jpg', 'Gurgen Arakelov'),\
                         EncodingFace('andrey.jpg', 'Andrey Krivosheev')]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            for thr in list_thread_recognise:
                thr.compare_face(face_encoding)

            name = "Unknown"

            is_cont = True
            while is_cont:
                is_one_alive = False
                for thr in list_thread_recognise:
                    if(thr.is_alive()):
                        is_one_alive = True
                        break;
                if not is_one_alive:
                    is_cont = False
            for thr in list_thread_recognise:
                if thr.is_matched():
                    name = thr.get_name()

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
