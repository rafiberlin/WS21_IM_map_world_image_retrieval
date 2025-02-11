import numpy as np
import gensim
import torch
import transformers as ppb

class Vectorizer:

    def __init__(self, device=None, pretrained_weights='distilbert-base-uncased', to_numpy_array: bool = False):
        self.vectors = []

        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device

        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            pretrained_weights)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        self.model = model.to(self.device)
        self.to_numpy_array = to_numpy_array

    def encode(self, sentences, convert_to_tensor= True):
        """
        Returns the embedding reoresentation of the sentence. The second parameter is only used,
        because we want to be able to switch to the  Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
        model (https://www.sbert.net/)
        :param sentences:
        :param convert_to_tensor:
        :return:
        """
        self.bert(sentences)
        return self.vectors

    def bert(self, sentences):
        tokenized = list(map(lambda x: self.tokenizer.encode(x, add_special_tokens=True), sentences))

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = torch.tensor([i + [0] * (max_len - len(i)) for i in tokenized], dtype=torch.long)
        input_ids = padded.to(self.device)
        # attention_mask = torch.tensor(np.where(padded != 0, 1, 0)).type(torch.LongTensor)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids)

        vectors = last_hidden_states[0][:, 0, :]

        if self.to_numpy_array:
            vectors = vectors.cpu().numpy()

        self.vectors = vectors

    def word2vec(self, words, pretrained_vectors_path, ensemble_method='average'):
        model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_vectors_path)

        vectors = []
        for element in words:
            temp = []
            for w in element:
                temp.append(model[w])
            if ensemble_method == 'average':
                vectors.extend([np.mean(temp, axis=0)])

        self.vectors = vectors


