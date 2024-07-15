import torch
import random
import numpy as np
import torchvision.datasets
from tensorflow import summary
import datetime
import torch.autograd.profiler as profiler
import param as _param
import resize as _resize
import log_reg as _log_reg
import mlp1 as _mlp1
import mlp2 as _mlp2
import net_const_tensor as _net


def tests(test, test_label, loss, net, summ, date, batch_size, weight_decay, iterations, device):

  net.eval()

  accuracy = 0
  loss_mean = 0
  accuracy_balance = 0
  accuracy_class = torch.zeros(len(torch.bincount(test_label)))
  loss_reg = 0

  iter = np.ceil(len(test)/batch_size)

  for start_index in range(0, len(test), batch_size):

    x_batch = test[start_index:start_index+batch_size].to(device)
    y_batch = test_label[start_index:start_index+batch_size].to(device)

    preds = net.forward(x_batch)

    loss_value = loss(preds, y_batch).data.cpu()
    loss_mean += loss_value

    label_true = preds.argmax(dim=1) == y_batch
    accuracy += label_true.float().mean().data.cpu()

    labels =  torch.bincount(y_batch)
    pred_labels =  torch.bincount(y_batch[label_true])
    labels_no_zero = labels != 0
    labels_no_zero[len(pred_labels):] = False
    accuracy_balance_batch = torch.zeros(len(labels))
    accuracy_balance_batch[labels_no_zero] = torch.div(pred_labels[labels_no_zero[:len(pred_labels)]].type(torch.FloatTensor), labels[labels_no_zero].type(torch.FloatTensor))
    accuracy_balance_batch = accuracy_balance_batch.cpu()
    accuracy_class[:len(accuracy_balance_batch)] += accuracy_balance_batch

  accuracy /= iter
  loss_mean /= iter
  accuracy_class /= iter
  accuracy_balance = (accuracy_class.sum())/len(accuracy_class)

  #print(accuracy)

  if date == 'train':

    alpha = torch.tensor(weight_decay)
    l2_reg = None

    for param in net.parameters():
      if l2_reg == None:
        l2_reg = torch.norm(param)**2
      else:
        l2_reg += torch.norm(param)**2

    l2_reg *= alpha
    loss_reg = loss_mean + l2_reg
    loss_reg = loss_reg.detach()
    loss_reg = loss_reg.cpu()

  with summ.as_default():
    summary.scalar('loss', loss_mean, step=iterations)
    summary.scalar('accuracy', accuracy, step=iterations)
    summary.scalar('accuracy_balanced', accuracy_balance, step=iterations)
    if date == 'train':
      summary.scalar('loss_reg', loss_reg, step=iterations)

  return



def trains(net, device, optimizer, loss, train, train_label, test, test_label, batch_size, lr, weight_decay, momentum, epoches,
           train_summary_writer, test_summary_writer):

    iterations = 0

    for epoch in range(0, epoches):
        order = np.random.permutation(len(train))
        for start_index in range(0, len(train), batch_size):

            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index:start_index + batch_size]

            x_batch = train[batch_indexes].to(device)
            y_batch = train_label[batch_indexes].to(device)

            '''
            # profiler 
            #with profiler.profile(record_shapes=True) as prof:
                #with profiler.record_function("model_inference"):
                    #preds = net.forward(x_batch)
    
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
            #prof.export_chrome_trace("trace.json")
            '''

            preds = net.forward(x_batch)

            loss_value = loss(preds, y_batch)

            loss_value.backward()
            optimizer.step()

            net.eval()
            tests(test, test_label, loss, net, test_summary_writer, 'test', 128, weight_decay, iterations, device)
            if iterations % 100 == 0:
                tests(train, train_label, loss, net, train_summary_writer, 'train', 128, weight_decay, iterations,
                      device)

            iterations += 1

        # net=net.cpu()
        # torch.save(net, 'logreg/'+str(epoch)+'.pt')
        # net = net.to(device)

    net = net.cpu()
    return net


if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    date_train = torchvision.datasets.KMNIST('./', download=True, train=True)
    date_test = torchvision.datasets.KMNIST('./', download=True, train=False)

    train = torch.unsqueeze(date_train.test_data, dim=1).type(torch.FloatTensor)/255
    train_label = date_train.train_labels.type(torch.LongTensor)
    test = torch.unsqueeze(date_test.test_data, dim=1).type(torch.FloatTensor)/255
    test_label = date_test.test_labels.type(torch.LongTensor)

    '''
    # аугментация половины обучающей выборки
    order = np.random.permutation(len(train))
    t = tf(train[order[0]])
    ag_train = torch.unsqueeze(t, 1)
    ag_train_label = train_label[order[0]:order[0]+1]
    a = len(train)
    for index in range(1, int(a/2)):
      t = tf(train[order[index]])
      print((index/a)*200)
      ag_train = torch.cat((ag_train, torch.unsqueeze(t, 1)))
      ag_train_label = torch.cat((ag_train_label, train_label[order[index]:order[index]+1]))

    train = torch.cat((train, ag_train))
    train_label = torch.cat((train_label, ag_train_label))
    '''

    print('Total train: ', len(train_label))
    print('Total test: ', len(test_label))
    print('Total class: ', len(torch.bincount(test_label)))
    print('Total train class: ', torch.bincount(train_label))
    print('Total test class: ', torch.bincount(test_label))

    print(train.shape)

    #with tensorflow tensorboard
    #так как только с ним можно работать в Colab
    #запускать tensorboard --logdir = путь/logs/tensorboard
    current_time = str(datetime.datetime.now().timestamp())
    train_log_dir = 'logs/tensorboard/train/' + current_time
    test_log_dir = 'logs/tensorboard/test/' + current_time
    train_summary_writer = summary.create_file_writer(train_log_dir)
    test_summary_writer = summary.create_file_writer(test_log_dir)

    lr = _param.lr
    weight_decay = _param.weight_decay
    momentum = _param.momentum
    batch_size = _param.batch_size
    epoches = _param.epoches

    net = _log_reg.Logreg()
    #net = _net.Net()
    #net = _mlp1.MLP1()
    #net = _mlp2.MLP2()

    #net = torch.load('path/model.pt')

    print(net)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    net = trains(net, device, optimizer, loss, train, train_label, test, test_label, batch_size, lr, weight_decay, momentum, epoches, train_summary_writer, test_summary_writer)

    #tests(test, test_label, loss, net, test_summary_writer, 'test', 128, weight_decay, 0, device)



