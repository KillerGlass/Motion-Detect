import  numpy as np
import cv2


video = "videos/Cars.mp4"
out = "videos/Captura.avi"
out1 = "videos/Captura2.avi"
#/media/killer/Linux/projetos_visao/results

cap = cv2.VideoCapture(video)

hasFrame, frame = cap.read()

print(hasFrame, frame.shape)
#video writter gravar o video
fourcc = cv2.VideoWriter_fourcc(*('XVID'))#xvid formato avi
#sempre usar a maior dimensao primeiro, ultimo parametro é se o video eh colorido
writer = cv2.VideoWriter(out, fourcc, 25,(frame.shape[1],frame.shape[0]),False)
writer2 = cv2.VideoWriter(out1, fourcc, 25,(frame.shape[1],frame.shape[0]),True)

#conta os frames de um video
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#ter numeros aleatorios dentro de um intervalo, onde os numeros gerados tem a mesma probabilidade
#de serem selecionados, gerando uma distribuição uniforme
#print(np.random.uniform(size=25))

#desta forma pode se seleciona frames aleatorios de maneira uniforme da imagem,
# multiplica esse valor pelo numero total de frames
framesId = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

print(framesId)

'''cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
hasframe, frame = cap.read()
cv2.imshow("Teste", frame)
cv2.waitKey(0)'''

frames = []
for fid in framesId:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasframe, frame = cap.read()
    frames.append(frame)

print(hasframe)

'''for ex in frames:
    cv2.imshow("frame", ex)
    cv2.waitKey(0)'''

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
#print(medianFrame)
#visualiza(medianFrame)


cv2.imwrite("/media/killer/Linux/projetos_visao/results/model_median_frame.jpg",medianFrame)

cap.set(cv2.CAP_PROP_POS_FRAMES,0)#resetar para começar o frames do inicio, ele vai andando como se fosse um ponteiro
grayMedianFRame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
#visualiza(grayMedianFRame)

while(True):
    #ler os frames do video
    hasframe, frame = cap.read()

    if not hasframe:
        print('error')
        break

    #converte para filtro de cinza
    frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    writer2.write(frame)#grava esse frame no video

    #faz a subtração entre o frame cinza e o a mediana, faz a subtração do video pelo background
    dframe = cv2.absdiff(frameGray, grayMedianFRame)

    #binarizando o video, threshold realizar a binarização, otsu vai traça o limiar para binarizar
    th, dframe = cv2.threshold(dframe,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )

    writer.write(dframe)
    cv2.imshow('Frane',dframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cap.release()
writer2.release()



