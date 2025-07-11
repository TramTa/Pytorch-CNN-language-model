import torch
import string 
from . import utils


vocab = string.ascii_lowercase + ' .' 
    # abcdefghijklmnopqrstuvwxyz .

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError('Abstract function')


    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]



class Bigram(LanguageModel):     
    """
    Implements a simple Bigram model.  
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. 
    Use this to debug `language.py` functions.
    """

    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):      
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug `language.py` functions.
    """

    def predict_all(self, some_text):
        prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5*one_hot[0]
            prob[:-1, 1:] += 0.5*one_hot[1:]
            prob[0, 1:] += 0.5*one_hot[-1]
            prob[1:, 1:] += 0.5*one_hot[:-1]
        return (prob/prob.sum(dim=0, keepdim=True)).log()


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels=[32,64,128,256], out_channels=len(vocab), kernel_size=3, dilation=1):
            """
            Implement a convolution followed by a non-linearity (ReLU).            
            ===
            :param in_channels: Conv1d parameter
            :param out_channels: Conv1d parameter
            :param kernel_size: Conv1d parameter
            :param dilation: Conv1d parameter
            """
            
            super().__init__()
            layers = in_channels

            c = len(vocab)  
            L = []
            total_dilation = dilation  
            for l in layers:
                L.append(torch.nn.ConstantPad1d((2*total_dilation,0), 0))
                L.append(torch.nn.Conv1d(c, l, kernel_size, dilation=total_dilation))   
                L.append(torch.nn.ReLU())
                total_dilation *= 2
                c = l
            self.network = torch.nn.Sequential(*L)

        def forward(self, x):
            return self.network(x)  



    def __init__(self, in_channels=[32,64,128,256], out_channels=len(vocab), kernel_size=3, dilation=1): 
        super().__init__()
        self.network = self.CausalConv1dBlock(in_channels, out_channels, kernel_size, dilation)
        self.classifier = torch.nn.Conv1d(in_channels[-1], out_channels, 1)   


    def forward(self, x):
        """
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of ...one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        
        batch_size = x.shape[0]

        char_1st = torch.nn.Parameter( (1/28)*torch.ones((len(vocab), 1)) )[None,:,:]
        char_1st = char_1st.repeat(batch_size, 1, 1).to(device)
        
        if x.shape[2] == 0:
            return char_1st 
        
        output1 = self.classifier(self.network(x)).to(device)   # x = batch of one-hot 
        output2 =  torch.cat( (char_1st, output1), dim=2)  # logit 
        return output2       



    def predict_all(self, some_text):
        """
        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        """
        
        text_1hot = utils.one_hot(some_text)[None,:,:].cpu()   
        logit = self.forward( text_1hot ).cpu()  

        mm = torch.nn.LogSoftmax(dim=1)
        res = mm(logit).squeeze(0).detach()
        return res  



def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r




if __name__ == '__main__':
    ss = ''
    ss_1hot = utils.one_hot(ss)

    model = TCN()
    pred = model( utils.one_hot(ss)[None,:,:] )

    pred_a = model.predict_all(ss)
    pred_n = model.predict_next(ss)
    pred_n2 = pred_a[:, -1]
    pred_n3 = model.predict_all(ss)[:, -1]

    print('predict_all =', pred_a.shape, pred_a)
    print('predict_next =', pred_n.shape, pred_n)
    print('n2=', pred_n2)
    print('n3=', pred_n3)
