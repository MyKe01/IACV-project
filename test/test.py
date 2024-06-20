import cv2
from ultralytics import YOLO

# Carica il modello YOLOv8 pre-addestrato
model = YOLO('yolov5x.pt')

# Apri il video input
video_path = 'resources/tennis2full.mp4'
cap = cv2.VideoCapture(video_path)

# Controlla se il video si apre correttamente
if not cap.isOpened():
    print("Errore nell'apertura del video.")
    exit()

# Ottieni le proprietà del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definisci il codec e crea l'oggetto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Esegui il rilevamento sul frame
    results = model(frame)

    # Disegna le bounding box sul frame
    annotated_frame = results[0].plot()

    # Scrivi il frame annotato nel video output
    out.write(annotated_frame)

    # Visualizza il frame (opzionale)
    cv2.imshow('Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia i video capture e writer e chiudi tutte le finestre
cap.release()
out.release()
cv2.destroyAllWindows()
