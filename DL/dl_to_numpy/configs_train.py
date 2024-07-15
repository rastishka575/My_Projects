from easydict import EasyDict as edict


cfg = edict(
{
    'data':
    {
        'path': 'data/MNIST',
        'nrof_classes': 10,
    },

    'dataloader':
    {
        'type': 'train',
        'batch_size': 64,
        'nrof_classes': 10,
        'shuffle': True,
        'sample_type': 'default'
    },

    'hidden_nrof': 1,  # 1(best), 2
    'hidden_size': 200,  # 64, 128(best), 200
    'model':
    {
        'fc1':
        {
            'type': 'Linear',
            'class': 'function_linear',
            'init':
            {
                'input_shape': 28 * 28,
                'output_shape': 200,
                'use_bias': True,  # True(best), False
                'initialization_type': 'he',  # he(best), xavier, glorot
                'regularization': False,
                'weight_decay': 0.001
            }
        },
        'ac1':
        {
            'type': 'Relu',  # Relu(best), Sigmoid, Tanh
            'class': 'function_activation'
        },
        'fc_out':
            {
            'type': 'Linear',
            'class': 'function_linear',
            'init':
            {
                'input_shape': 200,
                'output_shape': 10,
                'use_bias': True,
                'initialization_type': 'he',
                'regularization': False,
                'weight_decay': 0.001
            }
            },
        'Softmax1':
        {
            'type': 'Softmax',
            'class': 'function_activation'
        }
    },
    'criterion': 'Cross_entropy',
    'optimizer':
    {
        'type': 'Sgd',  # Sgd(best), Msgd
        'learning_rate': 0.0001,
        'momentum': 0.9,
    },
    'epochs': 11,
    'accuracy': 'Accuracy',
    'transforms_append': False,
    'transforms':
    {
        'GaussianBlur':
        {
            'ksize': (5, 5)
        }
    },
    'experiment_name': 'train_base_model_new'
})

'''
train_base_model, train_weight_init_he, train_weight_init_xavier, train_weight_init_glorot,
train_optimizer_sgd, train_optimizer_msgd, train_function_activation_relu, train_function_activation_sigmoid,
train_function_activation_tanh, train_use_bias_true, train_use_bias_false, train_use_hidden_nrof_2,
train_use_hidden_nrof_1, train_hidden_size_128, train_hidden_size_64, train_base_model_with_transforms
'''