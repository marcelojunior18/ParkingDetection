import yaml
import numpy as np
import cv2

#recebe arquivos externos
fn = r"C:\Users\marcelo\Documents\opencv eclipse\opencvsub\video1.mp4"
fn_yaml = r"C:\Users\marcelo\Documents\opencv eclipse\opencvsub\CUHKSquareDemo.yml"

#diz o que vai ser usado nas ou nao nas operacoes
config = {'criar_ret_acima_estacionamento': True,#ativa desenho retangulo no video          
          'detectar_vaga': True,
          'detectar_vaga': True,
          'tamanho_necessario_contorno_ret': 150,
          'park_laplacian_th': 3.5,
          'tempo_det_vaga_ocupada': 3}

# captura a imagem 0 para wecom e fn recebe o arquivo
cap = cv2.VideoCapture(fn) #pegue o video do arquivo

# cria a variavel de subtracao de fundo que sera utilizada 
if config['detectar_vaga']:
    #onde faz a subtracao do fundo da imagem
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)    

# Ler arquivo de mapeamento
with open(fn_yaml, 'r') as stream:
    dados_estacionamento = yaml.load(stream)
contorno_estacionamento = []
limite_retangulo = []
mascara_estaciomento = []
#pecorrer os dados do arquivo .yml
for park in dados_estacionamento:
    #recebe o array do arquivo .yml
    pontos = np.array(park['pontos'])
    retangulo = cv2.boundingRect(pontos)
    mudar_pontos = pontos.copy()
    mudar_pontos[:,0] = pontos[:,0] - retangulo[0] 
    mudar_pontos[:,1] = pontos[:,1] - retangulo[1]
    contorno_estacionamento.append(pontos)
    limite_retangulo.append(retangulo)
    mask = cv2.drawContours(np.zeros((retangulo[3], retangulo[2]), dtype=np.uint8),
            [mudar_pontos], contourIdx=-1, color=255, thickness=-1, lineType=cv2.LINE_8)                            
    mask = mask==255
    mascara_estaciomento.append(mask)


# transformacao morfologica eliptica
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
# transformacao morfologica rentangular
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5,19))
status_estacionamento = [False]*len(dados_estacionamento)
buffer_estacionamento = [None]*len(dados_estacionamento)

# ler quadro por quadro 
while(cap.isOpened()):    
    # posicao atual do arquivo de video   
    video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 
    video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) 
    ret, frame = cap.read()    
    if ret == False:
        print("Capture Error")
        break    
    
    # Background Subtraction
    #faz a suavizacao das bordas usando gaussianblur
    frame_blur = cv2.GaussianBlur(frame.copy(), (5,5), 3)
    #converte o quadro para tons de cinza 
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    #Cria copia 
    frame_out = frame.copy()        

    if config['detectar_vaga']:
        fgmask = fgbg.apply(frame_blur)
        bw = np.uint8(fgmask==255)*255
        #diminui as bordas caso o valor seja um na parte interna    
        bw = cv2.erode(bw, kernel_erode, iterations=1)
        #Engrossar as formas na imagem atual aumenta e fecha a borda
        bw = cv2.dilate(bw, kernel_dilate, iterations=1)
        #retr_external recupera apenas o contorno exterior
        #procura contornos cnts e RETR_EXTERNAL contorno externo
        #cv2.CHAIN_APPROX_SIMPLE Ele remove todos os pontos redundantes e comprime o contorno,
        (_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop dos contorno estacionamento
        for c in cnts:
            # Se o contorno for menor que o declarado pequeno, nao sera feito o retangulo
            if cv2.contourArea(c) < config['tamanho_necessario_contorno_ret']:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 2)             
            
            
    if config['detectar_vaga']:        
        for ind, park in enumerate(dados_estacionamento):
            pontos = np.array(park['pontos'])
            retangulo = limite_retangulo[ind]
            roi_gray = frame_gray[retangulo[1]:(retangulo[1]+retangulo[3]),
                                   retangulo[0]:(retangulo[0]+retangulo[2])]            
            
            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
            pontos[:,0] = pontos[:,0] - retangulo[0] 
            pontos[:,1] = pontos[:,1] - retangulo[1]
            delta = np.mean(np.abs(laplacian * mascara_estaciomento[ind]))
            status = delta < config['park_laplacian_th']
            # Se for detectada uma mudanca no status de estacionamento, salve o tempo atual
            if status != status_estacionamento[ind] and buffer_estacionamento[ind]==None:
                buffer_estacionamento[ind] = video_cur_pos
            # Se o status ainda e diferente do salvo e o contador esta aberto
            elif status != status_estacionamento[ind] and buffer_estacionamento[ind]!=None:
                if video_cur_pos - buffer_estacionamento[ind] > config['tempo_det_vaga_ocupada']:
                    status_estacionamento[ind] = status
                    buffer_estacionamento[ind] = None
            # Se o status for o mesmo e o contador estiver aberto                   
            elif status == status_estacionamento[ind] and buffer_estacionamento[ind]!=None:                
                buffer_estacionamento[ind] = None                    
            #print("#%d: %.2f" % (ind, delta))
        #print(status_estacionamento)
        
    if config['criar_ret_acima_estacionamento']:                    
        for ind, park in enumerate(dados_estacionamento):
            pontos = np.array(park['pontos'])
            if status_estacionamento[ind]: color = (0,255,0)
            else: color = (0,0,255)
            cv2.drawContours(frame_out, [pontos], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)            
            moments = cv2.moments(pontos)        
            centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
            cv2.putText(frame_out, str(park['id']), (centroid[0]+1, centroid[1]+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0]-1, centroid[1]-1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0]+1, centroid[1]-1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0]-1, centroid[1]+1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), centroid, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)


    # Display video
    cv2.imshow('frame', frame_out)    
    cv2.imshow('background mask', bw)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        #salva a imagem como foto na pasta
        cv2.imwrite('frame%d.jpg' % video_cur_frame, frame_out)
    

cap.release()
cv2.destroyAllWindows()    