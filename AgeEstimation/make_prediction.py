
import sys
sys.path.append("/home/genesis/.local/install/caffe/python/")
import glob
import caffe
import numpy as np
from caffe.proto  import caffe_pb2
import random
import cv2


def prepareNetwork(net_deploy, net_model, mean_file):
    #meanfile okunmasi
    mean_blob = caffe_pb2.BlobProto()
    with open(mean_file) as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width)).mean(1).mean(1)

    #caffemodelin okunmasi
    net = caffe.Net(net_deploy, net_model, caffe.TEST)

    #resimlerin transform edilmesi
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean_array)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255)
    return [net, transformer]



caffe.set_mode_gpu()
mean_file = './models/mean.binaryproto'
net_deploy = './models/caffenet_deploy_1.prototxt'
net_model = './models/caffe_model_1_iter_10000.caffemodel'

net,transformer = prepareNetwork(net_deploy, net_model, mean_file)
predict_dir="./test/*.jpg"
test_img_paths = [img_path for img_path in glob.glob(predict_dir)]
random.shuffle(test_img_paths)
test_img_paths = test_img_paths[0:1000]
test_ids = []
preds = []
fage=0
age=0
tage=0
#tahmin yapma
for img_path in test_img_paths:
    try:
        img = caffe.io.load_image(img_path)
    except:
        print ('load image error:',img_path)
        continue

    #cv2.imshow("image",cv2.imread(img_path))
    #cv2.waitKey(0)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']
    i=0
    preds = preds + [pred_probas.argmax()]
    for date in img_path.split("_")[0:]:
        age=date[7:]
        #print(age)
        break
    #age=int(today)-int(birthday)
    if pred_probas.argmax()==0:
        if int(age)<20:
            tage+=1
        else:
            fage+=1
        print(img_path)
        print("Yasi:"+str(age))
        print("En iyi olasilik %" + str(format(float(pred_probas[0][0]) * 100, '.2f')) + " :0-19 yas gurubu")
        print pred_probas
    elif pred_probas.argmax()==1:
        if int(age) < 40 and int(age)>19:
            tage += 1
        else:
            fage += 1
       # print ("En iyi olasilik  %.2f :0-19:\n" % round(float(pred_probas[0][1])*100,2))
        print(img_path)
        print("Yasi:" + str(age))
        print("En iyi olasilik %"+str(format(float(pred_probas[0][1])*100,'.2f')) + " :20-39 yas gurubu")
        print pred_probas
    elif pred_probas.argmax()==2:
        if int(age) < 60 and int(age)>39:
            tage += 1
        else:
            fage += 1
        print(img_path)
        print("Yasi:" + str(age))
        print("En iyi olasilik %" + str(format(float(pred_probas[0][2]) * 100, '.2f')) + " :40-59 yas gurubu")
        print pred_probas[0][2]

    if pred_probas.argmax()==3:
        if int(age) > 60:
            tage += 1
        else:
            fage += 1
        print(img_path)
        print("Yasi:" + str(age))
        print("En iyi olasilik %" + str(format(float(pred_probas[0][3]) * 100, '.2f')) + " :60+ yas gurubu")
        print pred_probas
    print '-------'

print("Dogru Tahmin Sayisi:"+str(tage))
print("Yanlis Tahmin Sayisi:"+str(fage))
