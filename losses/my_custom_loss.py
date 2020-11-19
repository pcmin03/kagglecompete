
from .Distance_loss import * 
from .information_loss import *
from .Garborloss import *

def select_loss(args): 

    lossdict = dict()
    labelname = ""
    #suggest loss
  
    #compare loss
    if args.BCE == True:
        criterion = torch.nn.CrossEntropyLoss()
        labelname += 'CE_'

    lossdict.update({'mainloss':criterion})
        
    return lossdict, labelname