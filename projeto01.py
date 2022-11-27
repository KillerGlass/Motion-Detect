import numpy as np
import cv2


def obter_background_subtractor(tipo_bgs):
    if tipo_bgs == "MOG":
        bgs = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif tipo_bgs == "MOG2":
        bgs = cv2.createBackgroundSubtractorMOG2()
    elif tipo_bgs == "GMG":
        bgs = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif tipo_bgs == "KNN":
        bgs = cv2.createBackgroundSubtractorKNN()
    elif tipo_bgs == "CNT":
        bgs = cv2.bgsegm.createBackgroundSubtractorCNT()
    else:
        bgs = None
    return bgs


def obter_karnel(tipo_kernel):
    if tipo_kernel == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    elif tipo_kernel == "opening":
        kernel = np.ones((3, 3), np.uint8)
    elif tipo_kernel == "closing":
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = None
    return kernel


def obter_filtro(imagem, tipo_filtro):
    if tipo_filtro == "closing":
        filtro = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, obter_karnel("closing"), iterations=2)
    elif tipo_filtro == "opening":
        filtro = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, obter_karnel("opening"), iterations=2)
    elif tipo_filtro == "dilation":
        filtro = cv2.dilate(imagem, obter_karnel("dilation"), iterations=2)
    elif tipo_filtro == "combine":
        closing = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, obter_karnel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, obter_karnel("opening"), iterations=2)
        filtro = cv2.dilate(opening, obter_karnel("dilation"), iterations=2)
    else:
        filtro = None
    return filtro


video = cv2.VideoCapture("videos/Liverpool.mp4")
background = obter_background_subtractor("KNN")

while video.isOpened():
    f, frame = video.read()

    if not f:
        print("Fim")
        break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    mascara = background.apply(frame)
    mascara = obter_filtro(mascara, "combine")
    mascara = cv2.medianBlur(mascara, 5)
    # marcara = cv2.bilateralFilter(mascara, 9, 75, 75)

    contornos, hierarquia = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)

            fonte = cv2.FONT_HERSHEY_SIMPLEX
            corbranca = (255, 255, 255)
            corAzul = (255, 0, 0)

            cv2.rectangle(frame, (10, 25), (355, 55), corAzul, -1)
            cv2.putText(frame, "Movimento detectado!", (10, 50), fonte, 1, corbranca, 2, cv2.LINE_AA)

            cv2.drawContours(frame, cnt, -1, corAzul, 3)
            cv2.drawContours(frame, cnt, -1, corbranca, 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), corAzul, 3)

            """
            # Sobreposições transparentes
            for alpha in np.arange(0.8, 1.1, 0.9)[::-1]:
                frame_copy = frame.copy()
                output = frame.copy()
                cv2.drawContours(frame_copy, [cnt], -1, corAzul, -1)
                frame = cv2.addWeighted(frame_copy, alpha, output, 1 - alpha, 0, output)
            """

    resultado = cv2.bitwise_and(frame, frame, mask=mascara)
    cv2.imshow("mascara", mascara)
    cv2.imshow("frame", frame)
    cv2.imshow("resultado", resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
