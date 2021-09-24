import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as f

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
        self.embed = nn.Embedding(vocab_size , embed_size)
        self.lstm = nn.LSTM(embed_size , hidden_size , num_layers , batch_first = True)
        self.linear = nn.Linear(hidden_size , vocab_size)
    
    
    def forward(self, features, captions):
        
        embeding = self.embed(captions[:,:-1])
        embeding = torch.cat((features.unsqueeze(1) , embeding),1)
        hidd,_ = self.lstm(embeding)
        output = self.linear(hidd)
        return output

    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output_list = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          
            outputs = self.linear(hiddens)            
            _, prob = outputs.max(2)                        
            output_list.append(prob.item())
            inputs = self.embed(prob)                       
            
        return output_list
    
    
    
    
    
    
    
    