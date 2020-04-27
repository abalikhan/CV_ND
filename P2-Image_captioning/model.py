import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
#         #initialize hidden layer size
#         self.hidden_size = hidden_size
        
#         #initialize embed_size
#         self.embed_size = embed_size
        
#         #initialize vocab_size
#         self.vocab_size = vocab_size
        
#         #initialize number of layers 
#         self.num_layers = num_layers
        
        #embedded layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        #LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        #fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        #initializing the weights of final layer and LSTM units
        
        self.init_weights()
    
    def forward(self, features, captions):
        
        # embedding the captions 
#         print(captions[:,:-1])
              
        word_embedding = self.embed(captions[:,:-1])
        
        
        #concatinate the features and captions embedding
        inputs = torch.cat((features.unsqueeze(1), word_embedding), dim=1)
        
        #apply the LSTM
        lstm_out, _ = self.lstm(inputs)
        
        #pass to an FC layer
        lstm_out = self.fc(lstm_out)
        
        return lstm_out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.fc(out.squeeze(1))
            _, predicted = out.max(1) 
            tokens.append(predicted.item())
            
            inputs = self.embed(predicted) 
            inputs = inputs.unsqueeze(1)
            
        return tokens
    
        
        "#initialize the hidden layers (This code taken from https://github.com/Bjarten"
        "for initialization of the model, I found this one of the best way to initialize weights for LSTM models)"
   
    def init_weights(self):
        ''' Initialize weights for fully connected layer and lstm forget gate bias'''
        
        # Set bias tensor to all 0.01
        self.fc.bias.data.fill_(0.01)
        # FC weights as xavier normal
        torch.nn.init.xavier_normal_(self.fc.weight)
        
        # init forget gate bias to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)