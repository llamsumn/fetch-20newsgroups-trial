from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet

class NLP:  
    def __init__(self, data):
        self.data = data
        self.stop_words = set(stopwords.words('english'))

        self.lowered = self.lowercase(self.data)
        self.punctuated = self.remove_punctuation(self.lowered)
        self.cleared = self.remove_hidden_characters(self.punctuated)
        self.blacked = self.whitespace_removal(self.cleared)
        self.tokenised = self.tokenise(self.blacked)
        self.cleaned = self.remove_stopwords(self.tokenised)
        self.texted = self.remove_numeric_tokens(self.cleaned)
        self.completed = self.remove_short_tokens(self.cleaned)

        self.stemmed = self.preprocess(self.completed, reduction='s')
        self.lemmatised = self.preprocess(self.completed, reduction='l')

        self.vocabulary_stemmed = self.build_vocabulary(self.stemmed)
        self.vocabulary_lemmatised = self.build_vocabulary(self.lemmatised)
        
    def lowercase(self, texts):
        return [t.lower() for t in texts]
    
    def remove_punctuation(self, texts):
        cleaned_texts = []
        for t in texts:
            t = t.translate(str.maketrans('', '', punctuation))
            cleaned_texts.append(t)
        return cleaned_texts
    
    def remove_hidden_characters(self, texts):
        cleaned_texts = []
        for t in texts:
            cleaned_texts.append(t.replace("\n", " ").replace("\t", " ").replace("\'", ''))
        return cleaned_texts
    
    def whitespace_removal(self, texts):
        return [t.strip() for t in texts]
    
    def tokenise(self, texts):
        return [word_tokenize(t) for t in texts]
    
    def remove_stopwords(self, texts):
        cleaned_texts = []
        for text in texts:
            cleaned_texts.append([w for w in text if not w in self.stop_words])
        return cleaned_texts
    
    def remove_numeric_tokens(self, texts):
        return [[w for w in text if not w.isdigit() and not any(char.isdigit() for char in w)] 
                for text in texts]
    
    def remove_short_tokens(self, texts, min_length=2):
        return [[w for w in text if len(w) >= min_length] for text in texts]
    
    def stem_texts(self, texts):
        stemmer = PorterStemmer()
        return [[stemmer.stem(word) for word in text] for text in texts]
    
    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  

    def lemmatise_texts(self, texts):
        lemmatiser = WordNetLemmatizer()
        lemmatised_texts = []
        for text in texts:
            pos_tags = pos_tag(text)
            lemmatised = [lemmatiser.lemmatize(word, self.get_wordnet_pos(pos)) 
                        for word, pos in pos_tags]
            lemmatised_texts.append(lemmatised)
        return lemmatised_texts
        
    def preprocess(self, texts, reduction):
        if reduction == 's':
            texts = self.stem_texts(texts)
        elif reduction == 'l':
            texts = self.lemmatise_texts(texts)
        return texts
    
    def build_vocabulary(self, texts):
        vocabulary = {}
        for text in texts:
          for word in text:
            if word not in vocabulary:
              vocabulary[word] = len(vocabulary)
        return vocabulary