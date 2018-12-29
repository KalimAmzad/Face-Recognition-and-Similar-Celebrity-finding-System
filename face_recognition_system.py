import face_recognition

# Load the known images
image_of_person_1 = face_recognition.load_image_file("ifthi_1.jpg")
image_of_person_2 = face_recognition.load_image_file("kalim_1.jpeg")
image_of_person_3 = face_recognition.load_image_file("tasin_1.jpg")
image_of_person_4 = face_recognition.load_image_file("zia_sir1.jpg")

# Get the face encoding of each person. This can fail if no one is found in the photo.
person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]
person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]
person_4_face_encoding = face_recognition.face_encodings(image_of_person_4)[0]

# Create a list of all known face encodings
known_face_encodings = [
    person_1_face_encoding,
    person_2_face_encoding,
    person_3_face_encoding,
    person_4_face_encoding
]

# Load the image we want to check
unknown_image = face_recognition.load_image_file("kalim_2.jpg")

# Get face encodings for any people in the picture
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)

    name = "Unknown"

    if results[0]:
        name = "Ifthi"
    elif results[1]:
        name = "Kalim"
    elif results[2]:
        name = "Tasin"
    elif results[3]:
        name = "Zia Sir"

    print(f"Found {name} in the photo!")
