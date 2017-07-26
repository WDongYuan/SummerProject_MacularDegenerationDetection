from sklearn.ensemble import RandomForestClassifier
import FeatureExtraction
import os
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC




if __name__=="__main__":
	path = "/Users/weidong/GoogleDrive/CMU/Summer Project/MyCode/BoundaryExtraction/image_dir/"
	file_list = os.listdir(path)
	image_list = []
	for file_name in file_list:
		if "jpg" in file_name:
			image_list.append(file_name)


	feature = []
	label = []

	print("Extracting feature.")
	count = 0
	# file = open("./feature.txt","w+")
	for image_name in image_list:
		count +=1
		if count%100==0:
			print(count)

		flag,tmp_feature = FeatureExtraction.GetFeature(path+"/"+image_name)
		if flag==False:
			continue

		# file.write(image_name+" ")
		# for one_feature in tmp_feature:
		# 	file.write(str(one_feature)+" ")
		# file.write("\n")

		feature.append(tmp_feature)
		if "WET" in image_name:
			label.append(0)
		else:
			label.append(1)
	# file.close()

	print("Training Random Forest...")
	x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)
	my_rf = RandomForestClassifier(n_estimators=50,max_depth=5)
	my_rf.fit(x_train, y_train)

	print("Evaluating...")
	print(my_rf.score(x_test,y_test))

	print("Training SVM Classifier...")
	my_svc = SVC()
	my_svc.fit(x_train,y_train)

	print("Evaluating...")
	print(my_svc.score(x_test,y_test))





