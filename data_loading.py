import pandas as pd
import numpy as np

class Data_Loading():
    """
    Load data from the raw text files from the following list:
    files = [
            "affectivetext_train_headline.txt", "affectivetext_train_emotion.txt", "affectivetext_train_valence.txt",
            "affectivetext_test_headline.txt", "affectivetext_test_emotion.txt", "affectivetext_test_valence.txt",
        ].
    Once data is loaded from the list of files, you may use the file affectivetext_data.csv.
    """

    def get_dataframe(load=False):
        """
        Get the dataframe containing the data from the Affective Text dataset.
        In the original dataset, multiple files are used containing training and test data.
        For use of this function, argue load = True in order to obtain a CSV file containing
        the compiled data if the six original text files are being used for the first time.

        :return: A dataframe containing all the Affective Text Data
        """
        if load:
            Data_Loading._write_data_to_csv()

        df = pd.read_csv("affectivetext_data.csv")
        return df.reindex(columns=["headline",
                                   "anger", "disgust", "fear", "joy", "sadness", "surprise",
                                   "valence"])

    def _write_data_to_csv():
        """
        To be run if user has raw data in form of text files in the following list:
        files = [
            "affectivetext_train_headline.txt", "affectivetext_train_emotion.txt", "affectivetext_train_valence.txt",
            "affectivetext_test_headline.txt", "affectivetext_test_emotion.txt", "affectivetext_test_valence.txt",
        ].
        A CSV file is created with the data extracted from the text files.
        """
        files = [
            "affectivetext_train_headline.txt", "affectivetext_train_emotion.txt", "affectivetext_train_valence.txt",
            "affectivetext_test_headline.txt", "affectivetext_test_emotion.txt", "affectivetext_test_valence.txt",
        ]
        # training data shape (3 x 1000), test data shape (3 x 250)
        training_data, test_data = [], []

        lines = []
        for i in range(6):
            with open(files[i]) as f:
                lines = [line.rstrip() for line in f]
            if i % 3 == 1:
                lines = [list(map(int, line.split())) for line in lines]
            if i % 3 == 2:
                lines = [int(line) for line in lines]

            if i <= 2:
                training_data.append(lines)
            if i > 2:
                test_data.append(lines)

        texts = np.array(training_data[0] + test_data[0]) # shape (1250,)
        texts = np.array([texts]) # shape (1,1250)
        emotions = np.array(training_data[1] + test_data[1]).T # shape (6,1250)
        valence = np.array(training_data[2] + test_data[2]) # shape (1250,)
        valence = np.array([valence]) # shape (1,1250)

        df = np.concatenate((valence, emotions, texts)).T # shape (1250,8)
        pd.DataFrame(df).to_csv("affectivetext_data.csv",
                                header=["valence",
                                        "anger", "disgust", "fear", "joy", "sadness", "surprise",
                                        "headline"],
                                index=False)

#### Original data preprocessing. Content beyond this point may be ignored as it was reformatted into the text files
#### specified above

# def clean_data():
#     file = "affectivetext_trial.valence.txt"
#     lines = []
#     with open(file) as f:
#         lines = f.readlines()
#
#     print(lines)
#
#     for i in range(0, len(lines)):
#         line = lines[i]
#         line = line[line.find(' ')+1:]
#         lines[i] = line
#
#     print(lines)
#     text = ''
#     for i in range(0, len(lines)):
#         text = text + lines[i]
#
#     f = open(file,'w')
#     f.write(text)
#     f.close()