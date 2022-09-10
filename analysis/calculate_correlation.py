from biomarkers import BioMarkers, ALL_MARKERS
import mat73
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GroupKFold,GridSearchCV,cross_val_score,cross_validate

# clf=LogisticRegression()
# gkf=GroupKFold(n_splits=5)
# param_grid = {'classifier__C': [0.01,0.05,0.1,0.5, 1,2,3,4,5,8, 10,12,15]}
# pipe=Pipeline([('scaler',StandardScaler()),('classifier',clf)])
# gscv=GridSearchCV(pipe,param_grid,cv=gkf,n_jobs=16)
# gscv.fit(features,label_array,groups=group_array)

# importance = clf.coef_[0]
# #importance is a list so you can plot it.
# feat_importances = pd.Series(importance)
# feat_importances.nlargest(20).plot(kind='barh',title = 'Feature Importance')

def calculateEEGPearson(eegData, labels):
    arrax = np.mean(eegData, axis=1) # the eegData is in shape (128, 12288, 13)
    pearson_corr = []
    for i in range(arrax.shape[0]):
        corr = pearsonr(arrax[i,:], labels)
        print(f"channel {i} pearson correlation {corr[0]}")
        pearson_corr.append(corr[0])

    plt.plot(pearson_corr)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="the path to the .mat file")
    args = parser.parse_args()

    raw_data = mat73.loadmat(args.file_name)
    signal = raw_data["Signal"]

    print(f"Complete loading {len(signal)} markers")
    markers = BioMarkers(signal)

    while True:
        print(
            "[1] Blood Pressure [2] ECG [3] EEG [4] EGG [5] EMG [6] EOG [7] GSR [8] Respitory [9] TREV [10] Exit"
        )
        num = int(input("Enter the number of the marker to calculate:"))
        if num == 10:
            exit()
        try:
            y = markers.get_labels()
            data, field = markers.get_data(ALL_MARKERS[num - 1])

            if num == 3 and field == 'data':
                calculateEEGPearson(data, y)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()