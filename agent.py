
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import pickle

from sklearn.preprocessing import LabelEncoder


class Encoder(object):
    """the tool of sklearn didn't work for me
        create a dict tha associate actions to numbers

    """
    def __init__(self):
        self.keys = dict()
        self.values = list()

    def fit(self,data):
        for x in data:
            if not tuple(x) in self.keys.keys():
                self.keys[tuple(x)]= len(self.values)
                self.values.append(tuple(x))
        return self

    def encode(self, data):
        return [self.keys[tuple(d)] for d in data] 

    def decode(self, data):
        return [self.values[int(d)] for d in data]




class ML(Agent):
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    datafile = "ml/datasave.dump"
    modelfile =  "ml/modelML2.dump"
    def __init__(self, environment):
        self.environment =environment
        self.load_data(ML.datafile) #comment lines fi you want use new datas 'or addd a security based on datetime of files..."'
        self.save(ML.modelfile)
        self.load_model(ML.modelfile)

    def act(self, observation):
        """Produces an action given an observation of the environment."""
        obs = observation.as_array()
        act = self.enc.decode(self.model.predict([obs]))[0]
        ret = self.environment.action_space.get_do_nothing_action()

        for i in range(len(ret)):
            ret[i] = act[i]
        return ret

    def train(self, dataset):
        self.datas = dataset
        X, y = list(zip(*dataset))
        self.enc = Encoder().fit(y)
        y= self.enc.encode(y)
        self.model = MLPClassifier((100,80, 80), activation="relu")
        #model = AdaBoostClassifier(n_estimators= 100)
        #odel = RandomForestClassifier(n_estimators= 500, )

        #self.model = RandomForestClassifier(n_estimators= 50)#attention tres lent a l'aprentissage
        for i in range(8):
            self.model.fit(X,y)
        

    def save(self, file):
        pickle.dump((self.model,self.enc), open(file, 'wb'))

    def load_model(self, file):
        self.model, self.enc = pickle.load(open(file, 'rb'))


    def load_data(self, file):
        datas = pickle.load(open(ML.datafile, "rb"))
        self.train(datas)





class TrainerAgent(Agent):
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    def __init__(self, environment):
        """Initialize a new agent."""
        self.agent = GreedySearch(environment)
        self.randomS = Agent(environment)
        self.randomL= RandomLineSwitch(environment)
        self.actions = list()
        self.environment = environment

    def act(self, observation):
       
        if np.random.rand() < 0.3 :
            action = self.agent.act(observation)
            self.actions.append( (observation.as_array(), action.as_array())) 
        else :
            if np.random.rand() < 0.3:
                action = self.randomS.act(observation)
            else :
                action = self.randomL.act(observation)
        return action

    def __del__(self):
        old_actions = list()
        if os.path.exists(ML.datafile):
            old_actions = pickle.load(open(ML.datafile, "rb"))
        wr = old_actions + self.actions
        pickle.dump(wr, open(ML.datafile, 'wb'))



class MLdagger(Agent):
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    datafile = "ml/dagsave.dump"
    def __init__(self, environment):
        self.environment =environment
        #self.train([tuple(),tuple()])

    def act(self, observation):
        """Produces an action given an observation of the environment."""
        obs = observation.as_array()
        act = self.enc.decode(self.model.predict([obs]))[0]
        ret = self.environment.action_space.get_do_nothing_action()

        for i in range(len(ret)):
            ret[i] = act[i]
        return ret

    def train(self, dataset):
        self.datas = dataset
        X, y = list(zip(*dataset))
        self.enc = Encoder().fit(y)
        y= self.enc.encode(y)
        #model = MLPClassifier((100,80, 80, 80), activation="tanh")
        #model = AdaBoostClassifier(n_estimators= 100)
        #odel = RandomForestClassifier(n_estimators= 500, )

        self.model = RandomForestClassifier(n_estimators= 500, max_depth=20)#attention tres lent a l'aprentissage
        self.model.fit(X,y)
        

    def save(self, file):
        pickle.dump((self.model,self.enc), open(file, 'wb'))

    def load_model(self, file):
        self.model, self.enc = pickle.load(open(file, 'rb'))


    def load_data(self, file):
        datas = pickle.load(open(ML.datafile, "rb"))
        self.train(datas)








class DAgger(Agent):
    def __init__(self, environment):
        """
            inspired by, with no resets
            http://proceedings.mlr.press/v15/ross11a/ross11a.pdf
        """
        self.agent = GreedySearch(environment)
        self.randoma = RandomNodeSplitting(environment)
        self.actions = list()
        self.environment = environment
        self.beta =0.5
        self.batch = 200
        self.ml = MLdagger(environment) #can be anotheer agent
        self.iter = 0
        self.ml.train([(np.zeros(428), np.zeros(76))])

    def act(self, observation):
        self.iter+=1
        if self.iter == 0%500:
            self.re_train(self.actions)
        
        action = self.agent.act(observation)
        self.actions.append( (observation.as_array(), action.as_array())) 

        if np.random.rand() < self.beta :    
            return action
        else :
            return self.ml.act(observation)
        return action


    def re_train(self, data):
        self.ml.train(data)

    def __del__(self): 
        old_actions = list()
        #if os.path.exists(ML.datafile):
        #    old_actions = pickle.load(open(ML.datafile, "rb"))
        wr = old_actions + self.actions

        pickle.dump(wr, open("ml/dagger.dump", 'wb'))#ML.datafile











#pyTorch imitation
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from  torchvision.transforms import ToTensor

from tqdm import tqdm

DEVICE = "cpu"#"gpu"


class NNModel(nn.Module):
    """docstring for Generator"""
    def __init__(self,input_size, output_size):
        super(NNModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.clf = nn.Sequential(
            #nn.Linear(64 * 6 * 6, 1024),
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace= True),
            nn.Linear(1024, output_size),
        )
        

    def forward(self, x):
        #x = x.view(x.size(0), 64 * 6 * 6)
        x = self.clf(x)
        return x



class ImitationDataset(Dataset):
    """docstring for ImiationDataset"""
    def __init__(self, file, transform = None):
        super(ImitationDataset, self).__init__()
        datas = pickle.load(open(ImitationTorch.datafile, "rb"))
        X, y = list(zip(*datas))
        self.X = torch.Tensor(X)
        if transform :
            self.X = transform(self.X)
        
        self.enc = Encoder().fit(y)
        y = self.enc.encode(y)
        ll = len(self.enc.values)
        self.y= torch.zeros((len(y),ll))
        for i, el in enumerate(y):
            self.y[i,el] = 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        






class ImitationTorch(Agent):
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    datafile = "ml/datasave.dump"
    modelfile =  "ml/model"
    def __init__(self, environment):
        self.environment = environment
        self.load_and_train(ML.datafile) #comment lines if you want use new datas 'or addd a security based on datetime of files..."'
        self.save(ImitationTorch.modelfile)
        self.load_model(ImitationTorch.modelfile)
        self.model.eval()

    def act(self, observation):
        """Produces an action given an observation of the environment."""
        obs = observation.as_array()
        predict = self.model(torch.Tensor(obs))
        act = self.enc.decode([torch.argmax(predict)])[0]
        ret = self.environment.action_space.get_do_nothing_action()

        for i in range(len(ret)):
            ret[i] = act[i]
        return ret

    def train(self, dataset):
        
        self.model = NNModel(input_size = dataset.X[0].size(0), output_size = len(dataset.enc.values))

        #bild data loader
        trainset = DataLoader(dataset,batch_size=256,shuffle=True)
        self.enc= dataset.enc
        fit(10, self.model, trainset, device = DEVICE)


    def save(self, file):
        pickle.dump((self.enc,self.model.input_size, self.model.output_size), open(file+".dump", 'wb'))
        torch.save(self.model.state_dict(), file+'.save')

    def load_model(self, file):
        self.enc,input_size, output_size = pickle.load(open(file+".dump", 'rb'))
        self.model = NNModel(input_size, output_size)
        self.model.load_state_dict(torch.load(file+".save"))


    def load_and_train(self, file):
        datas = ImitationDataset(file) 
        self.train(datas)




#classification_loss = torch.nn.CrossEntropyLoss()#TBD
classification_loss = torch.nn.MSELoss()#TBD

def eval_batch(model, x, y, opt,train, device):
    if train:
        opt.zero_grad()
    out = model(x)

    #helpers
    loss = classification_loss(out, y)

    if train: 
        loss.backward()
        opt.step()

    #s_acc = accuracy(s, s_true.to(args.device))
    return loss.item()
    #    if t_true is not None :
    #        t_acc = accuracy(t_clf, t_true)
    #    else :
    #        t_acc= torch.tensor(0)
    #    return np.array([ S_loss.item(), C_loss.item(), G_loss.item(), s_acc.item(),  t_acc.item()])

def run_epoch(model, opt, dataset, train, device):
    loss = 0 #np.zeros(1)
    if train:
        model.train()
    else :
        model.eval()

    for x, y in tqdm(dataset):
        loss += eval_batch(model, x.to(device), y.to(device),opt, train,  device)/len(dataset)
    return loss
    
def fit(epochs, model, trainset,device = "cpu"):
    out = list()

    opt = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.005)

    for epoch in range(epochs):
        train_loss = run_epoch(model,opt, trainset, train = True, device= device)
        
        #out.append((train_loss, valid_loss)) 
    