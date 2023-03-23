import torch
import torch.nn as nn
import data_loader
from torch.utils.data import DataLoader, TensorDataset
from model import Net
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter  
# /* *********************GPU 设置************************* */
device = torch.device('cuda:0' if torch.cuda.is_availabel() else 'cpu')


# /* *********************超参数设置************************ */
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 500
LEARNING_RATE = 0.00001

# /* ***********************数据设置************************ */

# 数据路径设置
data_charge_path = "D:/data_charge"
data_free_path = "D:/data_free"
save_path = 'best_model.pth'
writer = SummaryWriter('log')
# 加载数据
x_train, y_train, x_test, y_test = data_loader.loader(data_free_path, data_charge_path)

# 扩展通道
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# 像素值归一化[0, 255]->[0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 将label转化为one-hot编码
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

y_train = nn.functional.one_hot(y_train, NUM_CLASSES)
y_test = nn.functional.one_hot(y_test, NUM_CLASSES)

# 转化为loader形式
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(x_train, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(x_test, BATCH_SIZE, shuffle=False)

# /* *********************实例化网络************************ */
model = Net()

# /* ******************优化器、损失设置********************** */
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters)
best_loss = float('inf')


# /* *************************训练*************************** */
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    # running_loss = 0.0
    # running_corrects = 0
    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))
            # 记录训练损失和准确率
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            writer.add_scalar('Train/Accuracy', accuracy, epoch * len(train_loader) + i)
    #     running_loss += loss.item() * inputs.size(0)
    #     running_corrects += torch.sum(preds == labels.data)

    # epoch_loss = running_loss / len(train_set)
    # epoch_acc = running_corrects.double() / len(train_set)

# /* *************************测试*************************** */

    model.eval()
    # test_running_loss = 0.0
    # test_running_corrects = 0

    with torch.no_grad:
        correct = 0
        total = 0
        test_loss = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 打印测试结果并记录测试损失和准确率
        test_loss /= len(test_loader)
        test_accuracy = correct / total
        print('Epoch [{}/{}], Test Loss: {:.4f}, Test Accuracy: {:.2%}'.format(epoch+1, NUM_EPOCHS, test_loss, test_accuracy))
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
        #     test_running_loss += test_loss.item() * test_inputs.size(0)
        #     test_running_corrects += torch.sum(test_preds == test_labels.data)

        # test_epoch_loss = test_running_loss / len(test_set)
        # test_epoch_acc = test_running_corrects.double() / len(test_set)

    




