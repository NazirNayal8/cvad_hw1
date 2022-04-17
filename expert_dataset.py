import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root, img_size=(224, 224), mode='imitation_learning'):
        super().__init__()
        self.data_root = data_root
        # Your code here
        self.mode = mode

        if mode not in ['imitation_learning', 'affordances']:
            raise Exception(f'Undefined Dataset Mode: {mode}')


        image_names = os.listdir(os.path.join(data_root, 'rgb/'))
        self.original_size = (600, 800)
        self.img_size = img_size
        self.num_samples = len(image_names)

        if mode == 'imitation_learning':
            self.speeds = np.zeros(self.num_samples)
            self.commands = np.zeros(self.num_samples)
            self.actions = np.zeros((self.num_samples, 3))
        else:
            self.lane_dist = np.zeros(self.num_samples)
            self.route_angle = np.zeros(self.num_samples)
            self.tl_dist = np.zeros(self.num_samples)
            self.tl_state = np.zeros(self.num_samples)
            self.commands = np.zeros(self.num_samples)

        for i, img in enumerate(image_names):
            label_name = img[:-4] + '.json'

            with open(os.path.join(data_root, 'measurements', label_name), 'r') as f:
                m = json.load(f)
            
            if mode == 'imitation_learning':
                self.speeds[i] = m['speed']
                self.commands[i] = m['command']
                self.actions[i] = np.array([m['throttle'], m['steer'], m['brake']])
            else: 
                self.lane_dist[i] = m['lane_dist']
                self.route_angle[i] = m['route_angle']
                self.tl_dist[i] = m['tl_dist']
                self.tl_state[i] = m['tl_state']
                self.commands[i] = m['command']

        self.images = [os.path.join(data_root, 'rgb', img) for img in image_names]

        self.transforms = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Resize(img_size),
        ])


    def imitation_learning_getitem(self, index):
        
        image = cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB)
        image = self.transforms(image)

        v = torch.tensor(self.speeds[index]).type(torch.float)
        c = torch.tensor(self.commands[index]).type(torch.long)
        actions = torch.from_numpy(self.actions[index]).type(torch.float)

        return image, v, c, actions

    def affordances_getitem(self, index):

        image = cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB)
        image = self.transforms(image)

        c = torch.tensor(self.commands[index]).type(torch.long)
        lane_dist = torch.tensor(self.lane_dist[index]).type(torch.float)
        route_angle = torch.tensor(self.route_angle[index]).type(torch.float)
        tl_dist = torch.tensor(self.tl_dist[index]).type(torch.float)
        tl_state = torch.tensor(self.tl_state[index]).type(torch.long)

        return image, c, lane_dist, route_angle, tl_dist, tl_state

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here

        if self.mode == 'imitation_learning':
            return self.imitation_learning_getitem(index)
        else:
            return self.affordances_getitem(index)

    def __len__(self):
        return self.num_samples

        
