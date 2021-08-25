import numpy as np
from torch.nn.modules.module import Module
import torch

class PruningModule(Module):
    DEFAULT_PRUNE_RATE = {
        'conv1': 72,
        'conv2': 14,
        'conv3': 8,
        'conv4': 10,
        'conv5': 8,
        'fc1': 2,
        'fc2': 0.5,
        'fc3': 2
    }

    def _prune(self, module, threshold):

        #################################
        # TODO:
        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        #################################
        dev = module.weight.device
        weights = module.weight.data
        mask = weights.abs().ge(threshold)
        val = torch.zeros(weights.size(), device = dev)
        val[mask] = weights[mask]
        module.weight.data = val
        
        

    def prune_by_percentile(self, q=DEFAULT_PRUNE_RATE):

        ########################
        # TODO
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the (100 - q)th percentile
        # 	of absolute W as the threshold, and then set the absolute weights less than threshold to 0 , and the rest
        # 	remain unchanged.
        ########################
        for name, module in self.named_modules():
            if name in q:
                # Calculate percentile value
                percent = (100-q[name])/100
                print('percent:', percent)
                weights = module.weight.data
                threshold = torch.quantile(weights.abs(), percent)
                # Prune the weights and mask
                print(f'Pruning with threshold : {threshold:.4f} for layer {name}')
                self._prune(module, threshold)

    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():

            #################################
            # TODO:
            #    Only fully connected layers were considered, but convolution layers also needed
            #################################
            # ,'conv1', 'conv2','conv3','conv4','conv5'
            if name in ['fc1', 'fc2', 'fc3', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold:.4f} for layer {name}')
                self._prune(module, threshold)





