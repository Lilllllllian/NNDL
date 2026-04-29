import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        eval_iters = kwargs.get("eval_iters", log_iters)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_score = 0
        global_iter = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                if hasattr(self.model, 'set_training'):
                    self.model.set_training(True)
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)

                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if global_iter % eval_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score

                if (global_iter) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    if len(self.dev_scores) > 0:
                        print(f"[Dev] loss: {self.dev_loss[-1]}, score: {self.dev_scores[-1]}")
                global_iter += 1
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        if hasattr(self.model, 'set_training'):
            self.model.set_training(False)
        total_loss = 0
        total_score = 0
        total_num = 0
        for iteration in range(int(X.shape[0] / self.batch_size) + 1):
            batch_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
            batch_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
            if batch_X.shape[0] == 0:
                continue
            logits = self.model(batch_X)
            loss = self.loss_fn(logits, batch_y)
            score = self.metric(logits, batch_y)
            total_loss += loss * batch_X.shape[0]
            total_score += score * batch_X.shape[0]
            total_num += batch_X.shape[0]
        return total_score / total_num, total_loss / total_num
    
    def save_model(self, save_path):
        self.model.save_model(save_path)