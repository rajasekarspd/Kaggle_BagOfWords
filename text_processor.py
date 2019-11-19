#Importing necessary packages
import re
import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextProcessor():
    """
    Initialize TextProcessor with NLP engine NLTK or spaCy.
    Parameters
    -----------
    engine = 'nltk' or 'spacy'
    
    List of Methods:
    ----------------
    1. TextProcessor.tokenizer() - Converts given text into tokens(word/sentence).
    2. TextProcessor.get_stopwords() - To get list of stopwords.
    3. TextProcessor.add_stopwords() - To extend stopwords collection.
    4. TextProcessor.remove_stopwords() - To remove stopwords from existing collection.
    5. TextProcessor.get_text_rootform() - To retrieve root form (Stemming/Lemmatization) of given text from sentence.
    6. TextProcessor.get_abbrev() - To get list of available abbrevation collection.
    7. TextProcessor.add_abbrev() - To extend abbrevation collection.
    8. TextProcessor.remove_abbrev() - To remove any abbrevations from collection.
    9. TextProcessor.replace_abbrev() - To replace abbrevation from given text.
    10. TextProcessor.clean_text() - To clean given text removing all impurities.
    11. TextProcessor.remove_stopwords_fromtext() - To remove stopwords from given sentence.
    """
    def __init__(self, engine='nltk'):
        
        #Initializing default values        
        self.engine = engine.lower()
        engine_package = ['spacy','nltk']
        
        if self.engine not in engine_package:
            print('Package Error: Only spacy and nltk are supported currently.')
        elif self.engine=='spacy':
            print('Selected Engine:',self.engine)
            self.spacy_nlp = spacy.load('en_core_web_sm') #Loading Spacy NLP
            self.stop_words = list(spacy.lang.en.stop_words.STOP_WORDS) #Loading Spacy stopwords
        else:
            print('Selected Engine:',self.engine)
            self.stop_words = nltk.corpus.stopwords.words('english') #Loading nltk stopwords

        #abbrevations
        self.replacement_dicts = {r'\\n':' ', ' pt ': ' patient ', ' yo ': ' year old ', ' sw ': ' Social Worker ',
                             ' No t ': ' Not ',' unk ': ' unknown ', ' hx ': ' history ',' yrs ': ' years ', ' yr ': ' year ',                             ' couldn t ':' could not ',
                             ' can t ': 'can not ',  ' h o ':' history of ', ' w o ': ' with out ',' that\'s ':' that is ',
                             ' there\'s ':' there is ',' what\'s ':' what is ',' where\'s ':' where is ',' it\'s ':' it is ',
                             ' who\'s ':' who is ',' i\'m ':' i am ',' she\'s ':' she is ',' he\'s ':' he is ',
                             ' they\'re ':' they are ',' who\'re ':' who are ',' ain\'t ':' am not ',' wouldn\'t ':' would not ',
                             ' shouldn\'t ':' should not ',' can\'t ':' can not ',' couldn\'t ':' could not ',
                             ' won\'t ':' will not '
                             }                                
    
    def clean_text(self, input_text):
        """
        Processes the give text and removes all non words, digits, single letters and extra spaces.
        
        Parameters
        -----------
        1. input_text = Text to clean.
        2. token = 'word' or 'sentence'
        
        Returns: Text.
        
        """
        
        text = re.sub(r'\W',' ', input_text) #Remove all non words
        text = re.sub(r'\d+',' ', text) #Remove all digits
        text = text.lower() #Converting text into lowercase
        text = re.sub(r'\s+[a-z]\s+',' ', text) #Remove all single letters
        text = re.sub(r'^\s+','', text) #Remove space from start of text
        text = re.sub(r'\s+$','', text) #Remove space from end of text
        text = re.sub(r'\s+',' ', text) #Remove all multi space    
        return text
    
    def tokenizer(self,input_text, token='word'):
        """
        Converts given text into tokens.
        
        Parameters
        -----------
        1. input_text = Text to tokenize.
        2. token = 'word' or 'sentence'
        
        Returns: List of word or sentence. 
        
        """
        
        #Converting input into lower case for validation
        input_text = input_text.lower()
        token = token.lower()
                    
        if self.engine=='nltk' and token=='word':
            tokenized = nltk.word_tokenize(input_text)
        elif self.engine=='nltk' and token=='sentence':
            tokenized = nltk.sent_tokenize(input_text)
        elif self.engine=='spacy' and token =='word':
            tokenized = self.spacy_nlp(input_text)
            tokenized = [token.text for token in tokenized]
        elif self.engine == 'spacy' and token == 'sentence':
            tokenized = self.spacy_nlp(input_text)
            tokenized = [token for token in tokenized.sents]
        else:
            print('Tokenizer Error - nltk:(word|sentence) spacy:(word|sentence)')
            tokenized = 'Error'
            
        return tokenized
    
    def get_stopwords(self):
        """
        To get stopwords used in Text Processor.
        ---------------------------
        Parameters: None
        Returns: List of Stopwords.
        ---------------------------
        """        
        return list(self.stop_words)


    def remove_stopwords_fromtext(self, input_text):
        """
        To remove stopwords from given sentence.
        ---------------------------
        Parameters: sentence or paragraph
        Returns: Sentence.
        ---------------------------
        """                
        stopwords = list(self.stop_words)
        input_text = input_text.lower()
        
        if self.engine=='nltk':
            words = nltk.word_tokenize(input_text)
            words = [word for word in words if word not in stopwords]
            words = ' '.join(words)
        elif self.engine == 'spacy':
            words = self.spacy_nlp(input_text)
            words = [word.text for word in words if word.text not in stopwords]
            words = ' '.join(words)
        else:
            print('Remove stopwords from text: Error in input')
            words = 'Error in input.'
        
        return words
    

    def add_stopwords(self, new_stopwords):
        """
        Helps to extend stopwords collection in Text Processor package.
        
        Parameters
        -----------
        1. new_stopwords = List of new stopwords to add to existing collection of stopwords.
        
        Returns: None. 
        
        """        
                
        #If input is one word append it directly
        if isinstance(new_stopwords,str):
            old_stopwords = list(self.stop_words)
            old_stopwords.append(new_stopwords)

        else:
            new_stopwords = list(new_stopwords) #Transform old and new stopwords into lists
            old_stopwords = list(self.stop_words)
            old_stopwords.extend(new_stopwords) #Merge old and new stopwords
            old_stopwords = set(old_stopwords) #Convert to set and remove duplicates
                    
        self.stop_words = list(old_stopwords)
        
        return
    
    def remove_stopwords(self, rem_stopwords):
        """
        Helps to remove stopwords from existing collection.
        
        Parameters
        -----------
        1. rem_stopwords = List of stopwords to be removed.
        
        Returns: None. 
        
        """        
        #If input just one word
        if isinstance(rem_stopwords,str):
            old_stopwords = list(self.stop_words)
            old_stopwords.remove(rem_stopwords)            

        else:
            rem_stopwords = list(rem_stopwords) #Transform old and new stopwords into lists
            old_stopwords = list(self.stop_words)
            
            old_stopwords = [word for word in old_stopwords if word not in rem_stopwords]
                                            
        self.stop_words = list(old_stopwords)
        
        return
    
    def get_text_rootform(self, input_text, lemma_stemma=1):
        
        """
        Helps to retrieve root form of given text from sentence.  Performs text lemmatization or stemming from nltk and spacy libraries.
        
        Parameters
        -----------
        1. input_text = Text to perform lemmatization or stemming.
        2. lemma_stemma = 1 or 0 (1-lemma, 0-stemma)
        
        Returns: List of words from given sentence sliced to its root form. 
        
        """
        
        input_text = input_text.lower()
        words='Error in Root form input parameter.'
        
        if self.engine =='nltk' and lemma_stemma==0: #nltk stemming
            words = nltk.word_tokenize(input_text)
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
            words = ' '.join(words)
        
        elif self.engine == 'nltk' and lemma_stemma==1: #nltk lemmatization
            words = nltk.word_tokenize(input_text)
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            words = ' '.join(words)            
            
        elif self.engine == 'spacy' and lemma_stemma==1: #spacy lemmatization
            doc = self.spacy_nlp(input_text)
            words = [token.lemma_ for token in doc]
            words = ' '.join(words)            
        
        else:
            print('Rootform Error - nltk:(Lemmatize|Stemming), Spacy:(Lemmatize)')     
        return words   
    
    def get_abbrev(self):
        """
        Lists available abbrevation from Text Processor package.  Always returns updated collection.
        
        Parameter: None
        Returns: Dictionary
        
        """        
        return self.replacement_dicts
    
    def add_abbrev(self, input_dict):

        """
        Helps to extend abbrevation collection in Text Processor Package.
        
        Parameters
        -----------
        1. input_dict = A Dictionary with new abbrevations.
        
        Returns: None. 
        
        """        
        
        existing_abbrev = self.replacement_dicts
        
        if isinstance(input_dict, dict):
            for key in input_dict.keys():
                if key not in existing_abbrev.keys():
                    existing_abbrev[key]=input_dict[key]                    
            self.replacement_dicts = existing_abbrev                    
        else:
            print('Add Abbrevation Error: Input should be dictionary.')        
        return
    
    def remove_abbrev(self, input_dict):
        """
        Helps to remove abbrevations from existing collection inside TextProcessor Package.
        
        Parameters
        -----------
        1. input_dict = A Dictionary with abbrevations to get removed.
        
        Returns: None. 
        
        """        
        
        existing_abbrev = self.replacement_dicts
        
        if isinstance(input_dict, dict):
            for key in input_dict.keys():
                if key in existing_abbrev.keys():
                    existing_abbrev.pop(key)
            self.replacement_dicts = existing_abbrev                    
        else:
            print('Remove Abbrevation Error: Input should be dictionary.')                
        return
    
    def replace_abbrev(self, input_text):
        """
        To replace abbrevations from given text.
        
        Parameters
        -----------
        1. input_text = Sentence to replace abbrevations from
        
        Returns: text. 
        
        """                
        replace_dict = self.replacement_dicts
        
        for key in replace_dict.keys():
            input_text = re.sub(key, replace_dict[key], input_text)
        
        return input_text