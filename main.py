from logistic_regression_model import logistic_accu
from KNN_classification import knn_accu
from linear_svm_classification import linear_svm_accu
from kernel_svm_classification import rbf_svm_accu
from naive_bayes_classification import naive_accu
from decision_tree_classification import tree_accu
from random_forest_classification import forest_accu

if __name__ == '__main__':
    print(f'Logistic Regression\'s Accuracy Score-> {logistic_accu}')
    print(f'KNN Classification\'s Accuracy Score-> {knn_accu}')
    print(f'Linear Kernel SVM Classification\'s Accuracy Score-> {linear_svm_accu}')
    print(f'RBF Kernel SVM Classification\'s Accuracy Score-> {rbf_svm_accu}')
    print(f'Gaussian Naive Bayes Classification\'s Accuracy Score-> {naive_accu}')
    print(f'Decision Tree Classification\'s Accuracy Score-> {tree_accu}')
    print(f'Random Forest Classification\'s Accuracy Score-> {forest_accu}')
