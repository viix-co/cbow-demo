# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])




class CBOW(nn.Module):

    def __init__(self,vocab_size,embedding_dim):
        super(CBOW,self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.l = nn.Linear(embedding_dim,vocab_size)

    def forward(self, inputs):
        context_embedding=self.embeddings(inputs).view((-1,self.embedding_dim))    
        context_sum = torch.sum(context_embedding,dim=0).view((1,-1))
        l_out = self.l(context_sum)
        return F.log_softmax(l_out,dim=1)

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # example

losses=[]
criteria=nn.NLLLoss()
model=CBOW(vocab_size,EMBEDDING_DIM)
opt=optim.Adam(model.parameters(),lr=0.1)
for epoch in range(100):
    total_loss = 0.0
    for context,target in data:
        context_vector = make_context_vector(context,word_to_ix)
        probs = model(context_vector)
        model.zero_grad()
        loss = criteria(probs,torch.tensor([word_to_ix[target]],dtype=torch.long))
        loss.backward()
        opt.step()
        total_loss+=loss.item()
    losses.append(total_loss)
print(losses)