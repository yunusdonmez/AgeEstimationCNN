import sys
import glob
index=6
lastndx=0
print("Dosyalar Okunuyor...")
train_data = [img for img in glob.glob(sys.argv[1]+"/*.jpg")]
age0=0
age20=0
age40=0
age60=0
for in_idx, img_path in enumerate(train_data):
    for path in img_path.split("/")[-1:]:
        #print(path.split("_")[0])
        age=int(path.split("_")[0])
        if int(age)<20:
            label=0
            age0+=1
        elif int(age)<40:
            label=1
            age20+=1
        elif int(age)<60:
            label=2
            age40+=1
        else:
            label=3
            age60+=1
    print '{:0>5d}'.format(in_idx) + ':' + img_path+':'+str(label)



print '\nFinished processing all images'
print "0-19:"+str(age0)
print "20-39:"+str(age20)
print "40-59:"+str(age40)
print "60+:"+str(age60)