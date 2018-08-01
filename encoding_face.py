import threading
import face_recognition

class EncodingFace(threading.Thread):
    """
    the class for computing encoding faces
    into multithread mode
    """
    def __init__(self, name_file, vorname):
        threading.Thread.__init__(self)
        self.image = face_recognition.load_image_file(name_file)
        self.vorname = vorname
        self.image_face_encoding = face_recognition.face_encodings(self.image)[0]

    def compare_face(self, face_encoding):
        self.face_encoding = face_encoding
        self.run()

    def run(self):
        self.match = face_recognition.compare_faces([self.image_face_encoding], self.face_encoding)

    def is_matched(self):
        return self.match[0]

    def get_name(self):
        return self.vorname





