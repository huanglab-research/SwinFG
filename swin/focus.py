import torch
import math

# batch size
B = -1

import torch

# unify the meaning of the attention map
def get_map_mat():

    map_mats = {}
    window_Num = [8, 4, 2, 1]

    for window_n in window_Num:
        normal_map = torch.arange(0, window_n**2 * 12 * 12).view(window_n, window_n, 12, 12)
        
        shift_map = torch.zeros([window_n*12, window_n*12], dtype = torch.int)

        for patch_w in range(window_n):
            for patch_h in range(window_n):
                shift_map[patch_w*12:(patch_w+1)*12, patch_h*12:(patch_h+1)*12] = normal_map[patch_w][patch_h]
        temp = torch.roll(shift_map, (-6, -6), dims=(0, 1)) # 2D

        shfit_map = shift_map.view([window_n, window_n, 12, 12])

        for patch_w in range(window_n):
            for patch_h in range(window_n):
                shfit_map[patch_w][patch_h] = temp[patch_w*12:(patch_w+1)*12, patch_h*12:(patch_h+1)*12]

        shfit_map = shfit_map.view([-1])

        map_mat = torch.zeros([shfit_map.size()[0], shfit_map.size()[0]])
        for anum in range(shfit_map.size()[0]):
            map_mat[shfit_map[anum].item()][anum] = 1
        map_mats[window_n] = torch.inverse(map_mat.cuda())
    return map_mats

class focus():
    def __init__(self, args):
        self.store = []
        self.activ_size = [8, 8, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
        global B
        B = args.train_batch_size
        self.map_mats = get_map_mat()

        # Unify the size of each attention map
        self.pooling9216x9216to144x144 = torch.nn.AvgPool2d((64, 64), stride=(64, 64))
        self.pooling2304x2304to144x144 = torch.nn.AvgPool2d((16, 16), stride=(16, 16))
        self.pooling576x576to144x144 = torch.nn.AvgPool2d((4, 4), stride=(4, 4))

        self.affinity = torch.zeros([B, 144, 144])

        self.x_index = torch.arange(0, 12, dtype = torch.float).unsqueeze(1).cuda()
        self.y_index = torch.arange(0, 12, dtype = torch.float).cuda()

        self.region_map = None
        self.map_sum = None

        
    def en(self, data):
        self.store.append(data)

    def handle(self):

        for count, attn in enumerate(self.store):
            attn = attn.mean(1).view([B, -1, 144, 144])
            _, win_Count, _, _ = attn.size()

            attentionMap = torch.zeros([B, win_Count * 144, win_Count * 144]).cuda()
            
            # local window attention map to attention map
            for win_count in range(win_Count):
                    attentionMap[:, win_count * 144:(win_count+1)*144, win_count * 144:(win_count+1)*144] = attn[:, win_count]
     
            # unify the meaning of the attention map
            if count % 2 == 1:
                map_mat = self.map_mats[self.activ_size[count]]
                attentionMap = torch.matmul(torch.matmul(map_mat, attentionMap), map_mat.T)
            
            # unify the size of the attention map
            if self.activ_size[count] == 8:
                if count == 0:
                    self.affinity = self.pooling9216x9216to144x144(attentionMap)
                else:
                    self.affinity = torch.matmul(self.affinity, self.pooling9216x9216to144x144(attentionMap)) 
            elif self.activ_size[count] == 4:
                self.affinity = torch.matmul(self.affinity, self.pooling2304x2304to144x144(attentionMap)) 
            elif self.activ_size[count] == 2:
                self.affinity = torch.matmul(self.affinity, self.pooling576x576to144x144(attentionMap))      
            else:
                self.affinity = torch.matmul(self.affinity, attentionMap)                                     
    

        # region_map indicates the importance of the corresponding region of the image
        self.region_map = torch.matmul(self.affinity, self.affinity.permute(0, 2, 1)).sum(2).squeeze(1).view(-1, 12, 12)
        self.store = []

        # Find the coordinates of the center position
        self.map_sum = self.region_map.sum(1).sum(1)
        x_ = torch.matmul(self.region_map, self.x_index).sum(1) / self.map_sum.unsqueeze(1)
        y_ = torch.matmul(self.y_index, self.region_map).sum(1) / self.map_sum

        '''
        ------------------------ > x
        |- - - - - - - - - - - - -
        |- - - - - - - - - - - - -
        |- - - s * * * * - - - - -
        |- - - * * * * * - - - - -
        |- - - * * c * * - - - - -
        |- - - * * * * * - - - - -
        |- - - * * * * e - - - - -
        |- - - - - - - - - - - - -
        y
        The integrated attention map is shown in the figure c is the center position (x_, y_)
        s is the upper left corner of the judgment position (s_x, s_y) e is the lower right corner of the judgment position (e_x, e_y)
        '''

        # find the appropriate discriminative region according to the center
        for length in range(1, 11):
            s_x = x_ - length
            s_x[s_x < 0] = 0
            s_y = y_ - length
            s_y[s_y < 0] = 0
            e_x = x_ + length
            e_x[e_x > 11] = 11
            e_y = y_ + length
            e_y[e_y > 11] = 11

            # Find the percentage of the attention score in the currently acquired discriminative region
            mask = torch.zeros([B, 12, 12]).cuda()
            for temp in range(B):
                mask[temp:, int(s_x[temp]):int(e_x[temp]), int(s_y[temp]):int(e_y[temp])] = 1
            ratio = (self.region_map * mask).sum(1).sum(1) / self.map_sum

            if ratio.sum() / B > 0.6:
                break
    
        return x_, y_, s_x, s_y, e_x, e_y



