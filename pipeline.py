import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from optim import *
from concurrent.futures import ProcessPoolExecutor
from predictors.lce import LCEPredictor


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.bn2(x)

        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Pipeline:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = 64
        self.epochs = TOTAL_EPOCHS
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=4)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=4)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.predictor = LCEPredictor(metric="accuracy")
        self.criterion = nn.CrossEntropyLoss()

    def create_net(self):
        net = Net()
        return net.to(self.device)

    def create_optimizer(self, net, update_rule):
        optimizer = RLSearchOptimizer(net.parameters(), update_rule)
        return optimizer

    def create_scheduler(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer, self.epochs)
        return scheduler

    def train(self, net, optimizer):
        '''
        Evaluate the optimizer on a specific task.
        Note that both `epoch` and `batch` variables are numbered starting from 0 (0-indexed) within the code, 
        but in display they are represented as starting from 1 (1-indexed).
        Returns:
            useful (bool): If the optimizer incurs RuntimeError such as zero division, meaning that it is trivial program, in this case useful is False
            predicted_performance (float): The predicted performance of the optimizer on the task, in the original form (0-1). We should only use it when useful is True
        '''
        training_accs = []
        scheduler = self.create_scheduler(optimizer)
        for epoch in range(self.epochs):
            net.train()
            running_loss = 0.0
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

            for i, data in enumerate(self.trainloader):
                inputs, labels = data[0].to(
                    self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f'Epoch: {epoch+1:2d}, Batch: {i+1:4d}, Loss is NaN or Inf, terminating training.')
                    return False, None
                elif loss > 1e5:
                    print(
                        f"Epoch: {epoch+1:2d}, Batch: {i+1:4d}, Loss is unacceptably large: {loss:.4f}, terminating training.")
                    return False, None
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (i+1) % 100 == 0:
                    print(
                        f'Epoch: {epoch+1:2d}, Batch: {i+1:4d}, Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            scheduler.step()
            net.eval()
            correct, total = 0, 0
            total_loss = 0.0

            with torch.no_grad():
                for data in self.trainloader:
                    images, labels = data[0].to(
                        self.device), data[1].to(self.device)
                    outputs = net(images)
                    loss = self.criterion(outputs, labels)

                    total_loss += loss.item()*labels.shape[0]
                    predicted = torch.argmax(outputs, 1)
                    total += labels.shape[0]
                    correct += (predicted == labels).sum().item()

            training_loss = total_loss / total
            training_acc = correct / total
            training_accs.append(training_acc)
            print(
                f'Epoch: {epoch+1:2d}, Training loss: {training_loss:.4f}, Training acc: {training_acc:.4f}')

            # Prediction and Early Stopping
            if ENABLE_PREDICTION and (epoch+1) in TRIAL_EPOCHS:  # The elements in TRIAL_EPOCHS are 1-indexed
                # Move the data to CPU and transform it into a list
                learning_curves = torch.tensor(training_accs).cpu().tolist()
                future = self.executor.submit(
                    self.predictor.query, learning_curves)
                predicted_performance = future.result()
                predicted_performance = predicted_performance.item()
                print(f'Prediction completed: {predicted_performance}')
                # Early stop the evaluation for low performance programs
                if predicted_performance < EARLY_STOP_THRESHOLD:
                    print(
                        f'Early stopping at epoch {epoch+1} due to low predicted performance.')
                    # The predictor thinks that this optimizer is not worth training, early stop and return the last epoch acc
                    return True, training_acc
        # Not early stopped, return the final training accuracy as the reward
        return True, training_acc


def print_result(future):
    predicted_performance = future.result()
    print(f"[{datetime.now()}] Prediction completed: {predicted_performance}")
