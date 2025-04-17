import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import models

class SupervisedLearning():

    def __init__(self, train_loader, validation_loader, test_loader, model_name):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.model_name = model_name
        self.model = models.modeltype(self.model_name)
        self.model = self.model.to(self.device) # 이전 내 코드에서는 이게 밖에 있었어서 train함수 정해줄때 model을 따로 설정함

        print("Completed loading your network.")

        class_weights = torch.tensor([1.0, 1598 / 253], device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def evaluate_testset(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device).long()

                output = self.model(data)
                output = output.type(torch.float32)

                probs = torch.softmax(output, dim=1)

                _, pred = torch.max(probs, 1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm)

        # 저장 및 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("./save/confusion_matrix.png")
        plt.close()

        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        print(f"Test Accuracy: {acc:.4f}")

    def train(self, epochs, lr, l2): # epoch, lr, l2는 argparse에서 가져오기
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.995 ** epoch)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0  # epoch별 loss를 저장할 변수
            num_batches = 0  # 배치 개수 카운트

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device).long()

                optimizer.zero_grad() # 루프가 한번 돌고나서 역전파를 하기전에 반드시 zero_grad()로 .grad 값들을 0으로 초기화시킨 후 학습을 진행해야 한다.
                output = self.model(data)
                output = output.type(torch.float32)

                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() # loss 합산
                num_batches += 1

            avg_loss = total_loss / num_batches # 평균 loss 계산
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

            # validation 평가
            with torch.no_grad():
                self.model.eval()
                valid_accuracy = 0
                valid_loss = 0
                # autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함, inference시 불필요한 gradient계산 안함
                for idx, (x, y) in enumerate(self.validation_loader):
                    y = y.type(torch.LongTensor)
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    output = output.type(torch.float32)

                    loss = self.criterion(output, y)

                    probs = torch.softmax(output, dim=1)
                    _, pred = torch.max(probs, 1)
                    #prob, pred = torch.max(output.data, 1)
                    #pred = pred.type(torch.float32)
                    pred = pred.cpu().detach().numpy()
                    y = y.cpu().detach().numpy()

                    valid_loss += loss.item()
                    valid_accuracy += (pred == y.astype(np.uint8)).mean() # batch 단위 accuracy를 더함

                valid_accuracy /= len(self.validation_loader)
                valid_loss /= len(self.validation_loader)
                
                print(f"EPOCH:{epoch + 1}, Loss:{valid_loss}, Accuracy:{valid_accuracy}")

        # 모델 저장
        torch.save(self.model.state_dict(), "./save/efficient_chest_0417.pth")
        print("saved.....")

        self.evaluate_testset()