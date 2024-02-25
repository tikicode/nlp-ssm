import easydict
import nltk
from nltk.tokenize import word_tokenize  # for tokenization
import numpy as np # for numerical operators
import matplotlib.pyplot as plt # for plotting
import gensim.downloader # for download word embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm # progress bar
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Union
from easydict import EasyDict

# set random seeds
random.seed(42)
torch.manual_seed(42)

nltk.download('punkt')

def load_data() -> Tuple[Dict[str, List[Union[int, str]]], Dict[str, List[Union[int, str]]], Dict[str, List[Union[int, str]]]]:
    # download dataset
    print(f"{'-' * 10} Load Dataset {'-' * 10}")
    dataset = load_dataset("imdb")
    dataset = dataset.shuffle()  # shuffle the data
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    print(f"{'-' * 10} an example from the train set {'-' * 10}")
    print(train_dataset[0])

    print(f"{'-' * 10} an example from the test set {'-' * 10}")
    print(test_dataset[0])

    # cap the train and test sets at 20k and 1k, use the first 1k examples from the train set as the dev set
    # we convert each split into a torch dataset
    dev_dataset = train_dataset[:1000]
    train_dataset = train_dataset[1000:21000]
    test_dataset = test_dataset[:1000]

    return dev_dataset, train_dataset, test_dataset


def featurize(sentence: str, embeddings: gensim.models.keyedvectors.KeyedVectors) -> Union[None, torch.FloatTensor]:
    # sequence of word embeddings
    vectors = []

    # map each word to its embedding
    for word in word_tokenize(sentence.lower()):
        try:
            vectors.append(embeddings[word])
        except KeyError:
            pass

    if len(vectors) == 0: return None
    
    sum = np.sum(vectors, axis=0)
    avg = torch.from_numpy(sum / len(vectors))
    return avg


def create_tensor_dataset(raw_data: Dict[str, List[Union[int, str]]],
                          embeddings: gensim.models.keyedvectors.KeyedVectors) -> TensorDataset:
    all_features, all_labels = [], []
    for text, label in tqdm(zip(raw_data['text'], raw_data['label'])):
        feature = featurize(text, embeddings)
        if feature is not None:
            all_features.append(feature)
            all_labels.append(label)
    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return TensorDataset(features_tensor, labels_tensor)


def create_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool=True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SentimentClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(self.embed_dim, self.num_classes)
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inp):
        return self.linear(inp)

    @staticmethod
    def softmax(logits):
        c = torch.max(logits, dim=1, keepdim=True).values
        num = torch.exp(logits - c)
        return num / torch.sum(num, dim=1, keepdim=True)

    # The function that perform backward pass
    def gradient_loss(self, inp, logits, labels):
        bsz = inp.shape[0]
        probs = self.softmax(logits)
        onehot = F.one_hot(labels, num_classes=self.num_classes)

        grads_weights = ((inp.T @ (probs - onehot)) / bsz).T
        grads_bias = torch.sum(probs - onehot, dim=0) / bsz
        loss = F.cross_entropy(logits, labels) # tip from piazza

        return grads_weights, grads_bias, loss


# If your softmax implementation is correct, the following tests should pass
def test_softmax():
    test_inp1 = torch.FloatTensor([[1, 2], [1001, 1002]])
    test_inp2 = torch.FloatTensor([[3, 5], [-2003, -2005]])
    assert torch.allclose(SentimentClassifier.softmax(test_inp1),
                          torch.FloatTensor([[0.26894143, 0.73105860], [0.26894143, 0.73105860]]))
    assert torch.allclose(SentimentClassifier.softmax(test_inp2),
                          torch.FloatTensor([[0.11920292, 0.88079703], [0.88079703, 0.11920292]]))


# if your backward implementation is correct, the following tests should pass
def test_gradient_loss(model: SentimentClassifier):
    test_inp1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    test_logits1 = torch.FloatTensor([[0.3, -0.5], [-0.4, 0.6]])
    test_labels1 = torch.LongTensor([1, 1])
    gw1, gb1, loss1 = model.gradient_loss(test_inp1, test_logits1, test_labels1)
    assert torch.allclose(gw1, torch.FloatTensor([[ 0.8829,  1.3623,  1.8418],
        [-0.8829, -1.3623, -1.8418]]), atol=1e-4)
    assert torch.allclose(gb1, torch.FloatTensor([ 0.4795, -0.4795]), atol=1e-4)
    assert torch.abs(loss1 - 0.7422) < 1e-4


def accuracy(logits: torch.FloatTensor , labels: torch.LongTensor) -> torch.FloatTensor:
    assert logits.shape[0] == labels.shape[0]
    preds = torch.argmax(logits, 1)
    return (preds == labels).float()


def evaluate(model: SentimentClassifier, eval_dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    eval_losses = []
    eval_accs = []
    for batch in tqdm(eval_dataloader):
        inp, labels = batch
        logits = model(inp)
        # loss and accuracy computation
        _, _, loss = model.gradient_loss(inp, logits, labels)
        eval_losses.append(loss.item())
        eval_accs += accuracy(logits, labels).tolist()

    eval_loss, eval_acc = np.array(eval_losses).mean(), np.array(eval_accs).mean()
    print(f"Eval Loss: {eval_loss} Eval Acc: {eval_acc}")
    return eval_loss, eval_acc


def train(model: SentimentClassifier,
          learning_rate: float,
          train_dataloader: DataLoader,
          dev_dataloader: DataLoader,
          num_epochs: int,
          save_path: Union[str, None]=None):
    # record the training process and model performance for each epoch
    all_epoch_train_losses = []
    all_epoch_train_accs = []
    all_epoch_dev_losses = []
    all_epoch_dev_accs = []
    best_acc = -1.
    for epoch in range(num_epochs):
        model.train()
        print(f"{'-' * 10} Epoch {epoch}: Training {'-' * 10}")
        train_losses = []
        train_accs = []
        for batch in tqdm(train_dataloader):
            inp, labels = batch
            # forward and backward pass
            logits = model(inp)
            grads_weights, grads_bias, loss = model.gradient_loss(inp, logits, labels)
            # update parameters with gradient descent
            # we use torch.no_grad() to disable PyTorch-built in gradient tracking and calculation, details at (https://pytorch.org/docs/stable/generated/torch.no_grad.html)
            # since we are doing gradient descent manually
            with torch.no_grad():
                model.linear.weight -= grads_weights * learning_rate
                model.linear.bias -= grads_bias * learning_rate

            # record the loss and accuracy
            train_losses.append(loss.item())
            train_accs += accuracy(logits, labels).tolist()

        all_epoch_train_losses.append(np.array(train_losses).mean())
        all_epoch_train_accs.append(np.array(train_accs).mean())

        # evaluate on the dev set
        print(f"{'-' * 10} Epoch {epoch}: Evaluation on Dev Set {'-' * 10}")
        dev_loss, dev_acc = evaluate(model, dev_dataloader)
        # save the model if it achieves better accuracy on the dev set
        if dev_acc > best_acc:
            best_acc = dev_acc
            print(f"{'-' * 10} Epoch {epoch}: Best Acc so far: {best_acc} {'-' * 10}")
            if save_path:
                print(f"{'-' * 10} Epoch {epoch}: Saving model to {save_path} {'-' * 10}")
                torch.save(model.state_dict(), save_path)

        all_epoch_dev_losses.append(dev_loss)
        all_epoch_dev_accs.append(dev_acc)

    return all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs


def visualize_epochs(epoch_train_losses: List[float], epoch_dev_losses: List[float], save_fig_path: str):
    plt.clf()
    plt.plot(epoch_train_losses, label='train')
    plt.plot(epoch_dev_losses, label='dev')
    plt.xticks(np.arange(0, len(epoch_train_losses), 5).astype(np.int32)),
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_fig_path)


def run_grad_descent(config: easydict.EasyDict,
                     dev_data: Dict[str, List[Union[int, str]]],
                     train_data: Dict[str, List[Union[int, str]]],
                     test_data: Dict[str, List[Union[int, str]]]):
    # download and load embeddings
    # it might take a few minutes
    print(f"{'-' * 10} Load Pre-trained Embeddings: {config.embeddings} {'-' * 10}")
    embeddings = gensim.downloader.load(config.embeddings)

    # create datasets
    print(f"{'-' * 10} Create Datasets {'-' * 10}")
    train_dataset = create_tensor_dataset(train_data, embeddings)
    dev_dataset = create_tensor_dataset(dev_data, embeddings)
    test_dataset = create_tensor_dataset(test_data, embeddings)

    print(f"{'-' * 10} Create Dataloaders {'-' * 10}")
    train_dataloader = create_dataloader(train_dataset, config.batch_size, shuffle=True)
    dev_dataloader = create_dataloader(dev_dataset, config.batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_dataset,config.batch_size, shuffle=False)

    print(f"{'-' * 10} Load Model {'-' * 10}")
    model = SentimentClassifier(embeddings.vector_size, config.num_classes)

    print(f"{'-' * 10} Test Softmax {'-' * 10}")
    test_softmax()
    print(f"{'-' * 10} Pass Softmax Test {'-' * 10}")

    print(f"{'-' * 10} Test Backward {'-' * 10}")
    test_gradient_loss(model)
    print(f"{'-' * 10} Pass Backward Test {'-' * 10}")

    print(f"{'-' * 10} Start Training {'-' * 10}")
    all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs = (
        train(model, config.lr, train_dataloader, dev_dataloader, config.num_epochs, config.save_path))
    model.load_state_dict(torch.load(config.save_path))

    print(f"{'-' * 10} Evaluate on Test Set {'-' * 10}")
    test_loss, test_acc = evaluate(model, test_dataloader)

    return all_epoch_train_losses, all_epoch_train_accs, all_epoch_dev_losses, all_epoch_dev_accs, test_loss, test_acc


