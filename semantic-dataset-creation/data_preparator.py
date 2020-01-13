import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from extractor import DeepAutoencoder, DeepBeliefNetwork, ConvolutionalAutoencoder
import numpy as np

DATA_FOLDER = "data"

def prepare_data(data):
    max_len = max([len(x) for x in data])
    if max_len % 8:
        max_len += (8 - (max_len % 8))
    ret = np.array([np.pad(np.array(x), (0, max_len-np.array(x).shape[0]), 'constant') for x in data])
    return ret 


def main():

    FILE_NAMES = ["ant-1.5.csv", "ant-1.6.csv", "camel-1.2.csv", "camel-1.4.csv", "log4j-1.1.csv", "log4j-1.2.csv", "poi-2.0.csv", "poi-2.5.csv"]
    COUNT = 1
    
    for FILE_NAME in FILE_NAMES:
        name = FILE_NAME.split(".csv")[0]

        for num in range(COUNT):
            
            #Can be loaded from a different source
            X = np.load(DATA_FOLDER + "/" + name + "_X.npy")
            y = np.load(DATA_FOLDER + "/" + name + "_y.npy")
            
            X = prepare_data(X)
            y = np.array([np.array(x) for x in y])
            

            """ Extract features """
            for extractor in [DeepAutoencoder(),ConvolutionalAutoencoder(),DeepBeliefNetwork()]:
                print(extractor.__class__.__name__)
                X = extractor.get_features(X, y)
            
                X = np.array([x.flatten() for x in X])

                np.save(name +"_"+name+"_X_feat"+ "_" + str(num)+".npy", X) 

if __name__ == "__main__":
    main()