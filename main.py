import colorama
import re
import numpy as np
from colorama import Fore
import logistic_regression_model as lr
import KNN_classification as knn
import linear_svm_classification as l_svm
import kernel_svm_classification as rbf_svm
import naive_bayes_classification as naive
import decision_tree_classification as dtc
import random_forest_classification as rfc
colorama.init(autoreset=True)
model_list = [lr.classifier, knn.classifier, l_svm.classifier, rbf_svm.classifier, naive.classifier,
              dtc.classifier, rfc.classifier]
name_list = ["Logistic Regression", "KNN Classification", "Linear Kernel SVM Classification",
             "RBF Kernel SVM Classification", "Gaussian Naive Bayes Classification",
             "Decision Tree Classification", "Random Forest Classification"]
file = open('model_settings.txt', 'r')
model_name = int(file.read(1))
classifier = model_list[model_name-1]
print("------------------Breast Cancer Prediction------------------")
print(f"***Current Model:{name_list[model_name-1]}***")
file.close()
while True:
    print("1-Make a prediction\n"
          "2-Retrieve patient data\n"
          "3-Change prediction model\n"
          "4-Exit")
    option = input('Please select an action:')
    if option == '1':
        try:
            sample_number = int(input("Enter Sample Number:"))
        except:
            print(Fore.RED + 'Not a valid sample number!')
            sample_number = int(input("Enter Sample Number(Carefully!):"))
        print("Enter the following perimeters to make the prediction")
        x_0 = int(input('Clump thickness:'))
        x_1 = int(input('Uniformity of Cell Size:'))
        x_2 = int(input('Uniformity of Cell Shape:'))
        x_3 = int(input('Marginal Adhesion:'))
        x_4 = int(input('Single Epithelial Cell Size:'))
        x_5 = int(input('Bare Nuclei:'))
        x_6 = int(input("Bland Chromatin:"))
        x_7 = int(input("Normal Nucleoli:"))
        x_8 = int(input("Mitoses:"))
        X = np.array([[x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]])
        prediction = classifier.predict(X)
        print("--------------------Result--------------------")
        if prediction == 2:
            prediction = 'Benign Cell'
            print(Fore.GREEN + prediction)
        elif prediction == 4:
            prediction = 'Malignant Cell'
            print(Fore.RED + prediction)
        print("--------------------Result--------------------")
        f = open("Patient Record.txt", mode='a')
        f.writelines(
            [f"Patient Number: {sample_number}\n", f"Clump thickness: {x_0}\n", f"Uniformity of Cell Size: {x_1}\n",
             f"Uniformity of Cell Shape: {x_2}\n", f"Marginal Adhesion: {x_3}\n", f"Single Epithelial Cell Size: {x_4}\n",
             f"Bare Nuclei: {x_5}\n", f"Bland Chromatin: {x_6}\n", f"Normal Nucleoli: {x_7}\n", f"Mitoses: {x_8}\n",
             f"Result: {prediction}\n", "------------------------------------------------------------\n"
             ])
        print(Fore.GREEN + "***************Patient-Record-Updated***************")
        f.close()
    elif option == '2':
        f = open("Patient Record.txt", 'r')
        num = input('Patient Number:')
        n = 0
        while True:
            counter = 0
            f.seek(n + 16)
            data = f.read(6)
            f.seek(n)
            record = f.read(232)
            r = re.findall(r'10', record[22::])
            if len(r) != 0:
                record += f.read(len(r))
            if data == '':
                print('************************************************************************')
                print(colorama.Fore.RED + 'Invalid Patient Number!')
                print('************************************************************************')
                break
            if record[-11] == 'B':
                if data == num:
                    print('************************************************************************')
                    print(colorama.Fore.GREEN + record)
                    print('************************************************************************')
                    break
                else:
                    f.seek(0)
                    n = n + 306 + len(r)
            elif record[-11] == 'M':
                if data == num:
                    record = record + f.read(3)
                    print('************************************************************************')
                    print(colorama.Fore.RED + record)
                    print('************************************************************************')
                    break
                else:
                    f.seek(0)
                    n = n + 309 + len(r)
        f.close()
    elif option == '3':
        print("Below you can see the various model\'s score")
        print(f'1.Logistic Regression\'s Accuracy Score-> {lr.logistic_accu}')
        print(f'2.KNN Classification\'s Accuracy Score-> {knn.knn_accu}')
        print(f'3.Linear Kernel SVM Classification\'s Accuracy Score-> {l_svm.linear_svm_accu}')
        print(f'4.RBF Kernel SVM Classification\'s Accuracy Score-> {rbf_svm.rbf_svm_accu}')
        print(f'5.Gaussian Naive Bayes Classification\'s Accuracy Score-> {naive.naive_accu}')
        print(f'6.Decision Tree Classification\'s Accuracy Score-> {dtc.tree_accu}')
        print(f'7.Random Forest Classification\'s Accuracy Score-> {rfc.forest_accu}')
        model_num = input("Select the model number:")
        f = open('model_settings.txt', 'w')
        f.write(model_num)
        f.close()
        classifier = model_list[int(model_num)-1]
        print("************************************************************************")
        print(Fore.GREEN + f"Successfully changed the prediction model to {name_list[int(model_num)-1]}")
        print("************************************************************************")
    elif option == '4':
        print('Thanks for using this software!')
        exit()
    else:
        print('Invalid Option!')
        exit()
