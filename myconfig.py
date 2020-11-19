import argparse 

def my_config():
    parser = argparse.ArgumentParser(description='Process some integers')
    parser.add_argument('--knum', help='Select Dataset')
    parser.add_argument('--gpu', default='0',help='comma separated list of GPU(s) to use.',type=str)
    parser.add_argument('--weight_decay',default=1e-8,help='set weight_decay',type=float)
    parser.add_argument('--weight',default=100,help='set Adaptive weight',type=float)
    parser.add_argument('--start_lr',default=3e-3, help='set of learning rate', type=float)
    parser.add_argument('--end_lr',default=3e-6,help='set fo end learning rate',type=float)

    parser.add_argument('--scheduler',default='Cosine',help='select schduler method',type=str)
    parser.add_argument('--epochs',default=201,help='epochs',type=int)
    parser.add_argument('--out_class',default=37,help='set of output class',type=int)
    parser.add_argument('--changestep',default=10,help='change train to valid',type=int)
    parser.add_argument('--pretrain',default=False,help='load pretrained',type=bool)

    parser.add_argument('--datatype',default='uint16_wise', type=str)

    parser.add_argument('--Kfold',default=10,help='set fo end learning rate',type=int)
    #preprocessing 

    parser.add_argument('--batch_size', default=60,help='stride',type=int)
    parser.add_argument('--oversample', default=True, action='store_false',help='oversample')
    parser.add_argument('--use_train', default=False, action='store_true',help='make binary median image')
    parser.add_argument('--patchsize', default=256,help='stride',type=int)
    
    #loss
    
    parser.add_argument('--BCE',default=False, action='store_true',help='set Normalized Cross entropy')
    parser.add_argument('--cross_validation',default=True, action='store_false',help='set Normalized Cross entropy')
    parser.add_argument('--deleteall',default=False, action='store_true',help='set Adaptive_RMSE')
    parser.add_argument('--activation',default='sigmoid',type=str)
    
    parser.add_argument('--modelname',default='newunet_compare',help='select Garborloss',type=str)
    
    return parser.parse_args()