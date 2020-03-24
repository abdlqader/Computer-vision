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
        super(DecoderRNN,self).__init__()
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        
        #lstm define
        self.lstm = nn.LSTM(input_size=embed_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           #dropout=0.2,
                           batch_first=True)
        #embed vector define
        self.embed_vector = nn.Embedding(vocab_size,embed_size)
        #last linear layer from hidden to vocab
        self.linear = nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        #create vector for each word in our batch
        embedding = self.embed_vector(captions[:,:-1])
        
        inputs = torch.cat((features.unsqueeze(dim=1),embedding),dim=1)
        lstm_out,_ = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_capion = []
        
        for index in range(max_len):
            lstm_out, states = self.lstm(inputs, states)           #pass embeded features to our RNN
            outputs = self.linear(lstm_out)                        # 1,1,vocab_size
            outputs = outputs.squeeze(1)                           # 1,vocab_size
            wordid  = outputs.argmax(dim=1)                        # 1
            predicted_capion.append(wordid.item())
            inputs = self.embed_vector(wordid.unsqueeze(0))
        return predicted_capion