### Testing on YOLOv8x performance


import cv2
from ultralytics import YOLO

# Carica il modello YOLOv8 pre-addestrato
model = YOLO('yolov8x.pt')

# Apri il video input
video_path = 'resources/tennis2.mp4'
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

frame_count = 0
ball_detected_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Esegui il rilevamento sul frame
    results = model(frame)

    # Controlla se è stata rilevata una palla da tennis
    detected = False
    for result in results:
        for box in result.boxes:
            # Ottieni l'indice della classe e confrontalo con l'indice di 'sports ball'
            if model.names[int(box.cls)] == 'sports ball':
                detected = True
                break
        if detected:
            break

    if detected:
        ball_detected_count += 1

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

# Calcola la percentuale di frame in cui è stata rilevata una palla da tennis
if frame_count > 0:
    detection_percentage = (ball_detected_count / frame_count) * 100
    print(f"Total Frames: {frame_count}")
    print(f"Frames where a tennis ball was detected: {ball_detected_count}")
    print(f"Percentage of frames where a tennis ball was detected: {detection_percentage:.2f}%")
else:
    print("No frame processed.")
