import cv2

# Load gambar
img = cv2.imread('wajah.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tingkatkan kontras agar deteksi lebih akurat
gray = cv2.equalizeHist(gray)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Deteksi wajah (atur agar lebih sensitif)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Gambar kotak di sekitar wajah
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Tampilkan hasil
cv2.imshow('Deteksi Wajah', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
