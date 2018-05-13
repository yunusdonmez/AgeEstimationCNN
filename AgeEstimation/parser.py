import os
import shutil


files = os.popen("ls *.jpg | sort").read().split("\n")[:-1]
i=0
jpg_id=0
birthday="-"
today="-"
for filex in files:
	for date in filex.split("_"):
		if i==0:
			print("id="+date)
			jpg_id=date
			i=1
		elif i==1:
			for year in date.split("-"):
				birthday=year
				print("birthday="+birthday)
				i=2
				break
		else:
			today=date[0:-4]
			print("today="+today)
			i=0
	age=int(today)-int(birthday)
	shutil.move("./"+filex,"../dataset/"+str(age)+"_"+filex)