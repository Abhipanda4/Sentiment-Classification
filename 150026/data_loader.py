import os
import re
from nltk.corpus import stopwords

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.training_corpus = [] * 2
        self.stop_words = set(stopwords.words('english'))

    def read_lines(self, filename):
        with  open(filename) as fp:
            text = fp.read()
            text = re.sub('<[^<]+?>', '', text)
            text = text.split()
            new_text = [w for w in text if w not in self.stop_words]
            return " ".join(new_text)

    def gen_data(self):
        pos_samples = os.listdir(self.base_dir + "pos")
        neg_samples = os.listdir(self.base_dir + "neg")
        pos_reviews = []
        neg_reviews = []
        for txt_file in pos_samples:
            x = self.read_lines(self.base_dir + "/pos/" + txt_file)
            pos_reviews.append(x)
        for txt_file in neg_samples:
            x = self.read_lines(self.base_dir + "/neg/" + txt_file)
            neg_reviews.append(x)
        return pos_reviews, neg_reviews
