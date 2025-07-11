from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import string 
import torch
import heapq
import copy
import numpy as np 


vocab = string.ascii_lowercase + ' .' 
    # abcdefghijklmnopqrstuvwxyz .


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Evaluate the log-likelihood of a given string.

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """

    pred_all = model.predict_all(some_text)[:, :-1]    # predict all log prob: size (28, len_str) 
    text_1hot = utils.one_hot(some_text)
    return pred_all[ text_1hot==1. ].sum()


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """

    sen = '' 
    for i in range(max_length):
        char_prob = model.predict_next(sen).exp()   # prob of 28 char 
        idx = torch.distributions.Categorical(char_prob).sample()   # index of 1 char         
        sen += vocab[idx]
        if vocab[idx] == '.' :     
            break 
    return sen 



class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace

        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)

def log_likelihood_2(model: LanguageModel, some_text: str, average_log_likelihood=True):
    loglik = log_likelihood(model, some_text) 
    if len(some_text) == 0:
        return loglik

    if average_log_likelihood:
        loglik = loglik / len(some_text)
    return loglik 


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    
    if max_length == 0: 
        return ''   

    beam = TopNHeap(beam_size)
    beam2 = TopNHeap(beam_size)      

    for char in vocab:
        tmp = log_likelihood_2(model, char, average_log_likelihood) 
        beam.add( (tmp, char) )
        beam2.add( (tmp, char) )

    for k in range(28, beam_size):
        beam2.add( (-np.inf, 'a') )     

    for i in range(1, max_length):    
        beam_sen = copy.deepcopy( [item[1] for item in beam.elements] )     
        beam = TopNHeap(beam_size)      # new beam

        for sen in beam_sen:
            for char in vocab:
                if sen[-1] != '.'  and len(sen) < max_length:
                    sen_ext = sen + char 
                else: 
                    continue
                
                sen_loglik = log_likelihood_2(model, sen_ext, average_log_likelihood) 
                beam.add( (sen_loglik, sen_ext) )
        
        for k in range(len(beam.elements)):
            beam2.add( beam.elements[k] )
    res = sorted(beam2.elements, key=lambda x: x[0], reverse=True)

    return [ item[1] for item in res[:n_results] ]




# ========================= main
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['', 'abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.', 'dc', 'dce', 'bc', 'bca']:
        print(s, float(log_likelihood(lm, s)))
        print(s, float(log_likelihood_2(lm,s,True)))
    print()

    for s in beam_search(lm, 100):   
        print(s, float(log_likelihood(lm, s)) )  
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True, max_length= 100):   
        print(s, float(log_likelihood(lm, s)) )  

