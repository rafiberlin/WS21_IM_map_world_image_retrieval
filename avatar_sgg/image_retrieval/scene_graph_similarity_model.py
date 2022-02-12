import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from avatar_sgg.config.util import get_config


class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)
        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class ApplyAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(v_features, q_features, mid_features, drop))
        self.glimpse_layers = nn.ModuleList(layers)

    def forward(self, v, q, atten):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        atten:  batch x glimpses x v_num x q_num
        """
        for g in range(self.glimpses):
            atten_h = self.glimpse_layers[g](v, q, atten)
            q = q + atten_h
        # q = q * q_mask.unsqueeze(2)
        return q.sum(1)


class ApplySingleAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu', drop=drop)  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, activate='relu', drop=drop)
        self.lin_atten = FCNet(mid_features, mid_features, drop=drop)

    def forward(self, v, q, atten):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        atten:  batch x v_num x q_num
        """

        # apply single glimpse attention
        v_ = self.lin_v(v).transpose(1, 2).unsqueeze(2)  # batch, dim, 1, num_obj
        q_ = self.lin_q(q).transpose(1, 2).unsqueeze(3)  # batch, dim, que_len, 1
        # v_ = torch.matmul(v_, atten.unsqueeze(1)) # batch, dim, 1, que_len
        # This is the only way I found to make it match the dimension in the previous comment: # batch, dim, 1, que_len
        v_ = torch.matmul(v_.squeeze(2), atten.transpose(3, 1).squeeze(2)).unsqueeze(2)
        h_ = torch.matmul(v_, q_)  # batch, dim, 1, 1
        h_ = h_.squeeze(3).squeeze(2)  # batch, dim

        atten_h = self.lin_atten(h_.unsqueeze(1))

        return atten_h


class SGEncode(nn.Module):
    def __init__(self, img_num_obj=151, img_num_rel=51, txt_num_obj=4460, txt_num_rel=646):
        super(SGEncode, self).__init__()
        self.embed_dim = 512
        self.hidden_dim = 512
        self.final_dim = 1024
        self.num_layer = 2
        self.margin = 1.0
        self.img_num_obj = img_num_obj
        self.img_num_rel = img_num_rel
        self.txt_num_obj = txt_num_obj
        self.txt_num_rel = txt_num_rel

        self.img_obj_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_head_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_tail_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_pred_embed = nn.Embedding(self.img_num_rel, self.embed_dim)
        self.txt_obj_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_head_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_tail_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_pred_embed = nn.Embedding(self.txt_num_rel, self.embed_dim)

        self.apply_attention = ApplyAttention(
            v_features=self.embed_dim * 3,
            q_features=self.embed_dim,
            mid_features=self.hidden_dim,
            glimpses=self.num_layer,
            drop=0.2, )

        self.final_fc = nn.Sequential(*[nn.Linear(self.hidden_dim, self.hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hidden_dim, self.final_dim),
                                        nn.ReLU(inplace=True)
                                        ])

    def encode(self, inp_dict, is_img=False, is_txt=False):
        assert is_img + is_txt
        if len(inp_dict['relations'].shape) == 1:
            inp_dict['relations'] = torch.zeros(1, 3).to(inp_dict['entities'].device).long()
            inp_dict['graph'] = torch.zeros(len(inp_dict['entities']), 1).to(inp_dict['entities'].device).float()

        if is_img:
            obj_encode = self.img_obj_embed(inp_dict['entities'])
            rel_head_encode = self.img_rel_head_embed(inp_dict['relations'][:, 0])
            rel_tail_encode = self.img_rel_tail_embed(inp_dict['relations'][:, 1])
            rel_pred_encode = self.img_rel_pred_embed(inp_dict['relations'][:, 2])
        elif is_txt:
            obj_encode = self.txt_obj_embed(inp_dict['entities'])
            rel_head_encode = self.txt_rel_head_embed(inp_dict['relations'][:, 0])
            rel_tail_encode = self.txt_rel_tail_embed(inp_dict['relations'][:, 1])
            rel_pred_encode = self.txt_rel_pred_embed(inp_dict['relations'][:, 2])
        else:
            print('ERROR')

        rel_encode = torch.cat((rel_head_encode, rel_tail_encode, rel_pred_encode), dim=-1)

        atten = inp_dict['graph'].transpose(0, 1)  # num_rel, num_obj
        atten = atten / (atten.sum(0).view(1, -1) + 1e-9)

        sg_encode = self.apply_attention(rel_encode.unsqueeze(0), obj_encode.unsqueeze(0), atten.unsqueeze(0))

        return self.final_fc(sg_encode).sum(0).view(1, -1)

    def _to_tensor(self, inp_dict):
        return {'entities': torch.LongTensor(inp_dict['entities']),
                'relations': torch.LongTensor(inp_dict['relations'])}

    def __generate_tensor(self, graph_dict):
        img = self._to_tensor(graph_dict['img'])
        img_graph = torch.FloatTensor(graph_dict['image_graph'])
        img['graph'] = img_graph

        txt = self._to_tensor(graph_dict['txt'])
        txt_graph = torch.FloatTensor(graph_dict['text_graph'])
        txt['graph'] = txt_graph
        return img, txt


    def forward(self, graph_dict):
        """
        Modified from the original to only perform the inference.
        :param fg_imgs:
        :param fg_txts:
        :return:
        """

        assert type(graph_dict) is dict

        fg_img, fg_txt = self.__generate_tensor(graph_dict)
        fg_img_encode = self.encode(fg_img, is_img=True)
        fg_txt_encode = self.encode(fg_txt, is_txt=True)

        return torch.stack([fg_img_encode[0], fg_txt_encode[0]])

def get_scene_graph_encoder(pretrained_model_path=None):
    """
    Get the pretrained Scene Graph Encoder
    :param pretrained_model_path:
    :return:
    """
    model = SGEncode()

    if pretrained_model_path is None:
        pretrained_model_path = get_config()["scene_graph"]["pretrained_model_path"]

    checkpoint = torch.load(pretrained_model_path,  map_location=torch.device("cpu"))
    print("Loading pretrained model:", pretrained_model_path)
    model.load_state_dict(checkpoint)
    return model