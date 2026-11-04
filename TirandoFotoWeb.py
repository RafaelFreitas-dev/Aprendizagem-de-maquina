import cv2 

video = cv2.VideoCapture(0) 

classificadorFace = cv2.CascadeClassifier('ReconhecimentoDeteccao/haarcascade_frontalface_default.xml')

# Verifica se o classificador foi carregado corretamente
if classificadorFace.empty():
    raise FileNotFoundError(
        "Não foi possível carregar o classificador facial XML."
    )

while True: 
    conectado, frame = video.read() 
    #print(conectado) 
    #print(frame) 

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(70,70))

    for (x, y, l, a) in facesDetectadas: 
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 255, 50), 2) 
        cv2.imshow('Vídeo', frame) 
    if cv2.waitKey(1) == ord('q'): 
        break 

video.release() 

cv2.destroyAllWindows() 