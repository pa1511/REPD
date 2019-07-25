from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

standard_plot_width = 6
standard_plot_height = 5

def adjust_plot_size(plot,width=standard_plot_width,height=standard_plot_height):
    fig_size = plot.rcParams["figure.figsize"]
  
    fig_size[0] = width
    fig_size[1] = height
    plot.rcParams["figure.figsize"] = fig_size
    
def calculate_results(y_true,y_predicted):
    matrix = confusion_matrix(y_true, y_predicted)
    accuracy = accuracy_score(y_true, y_predicted)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_predicted, average='binary')

    return matrix, accuracy, precision, recall, f1_score

def print_confusion_matrix(matrix):
    print("Confusion matrix")
    print(matrix)

def print_results(accuracy, precision, recall, f1_score):
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)
    
