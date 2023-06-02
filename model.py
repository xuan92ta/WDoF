import torch
import torch.nn as nn


class WDoF(nn.Module):
    def __init__(self, opt, n_items):
        super(WDoF, self).__init__()

        self.z_dim = opt['z_dim']
        self.enc_dim = eval(opt['enc_dim'])
        self.dec_dim = eval(opt['dec_dim'])
        self.dropout_rate = opt['dropout_rate']
        self.n_items = n_items

        self.dropout = nn.Dropout(p=self.dropout_rate)

        # encoder initialization
        enc_layers = []
        for i in range(len(self.enc_dim)):
            if i == 0:
                enc_layers.append(nn.Linear(self.n_items, self.enc_dim[i]))
            else:
                enc_layers.append(nn.Linear(self.enc_dim[i-1], self.enc_dim[i]))
            enc_layers.append(nn.Tanh())
        enc_layers.append(nn.Linear(self.enc_dim[-1], self.z_dim*2))
        self.encoder = nn.Sequential(*enc_layers)
        self.encoder.apply(self.init_weights)

        # decoder initialization
        dec_layers = []
        for i in range(len(self.dec_dim)):
            if i == 0:
                dec_layers.append(nn.Linear(self.z_dim, self.dec_dim[i]))
            else:
                dec_layers.append(nn.Linear(self.dec_dim[i-1], self.dec_dim[i]))
            dec_layers.append(nn.Tanh())
        dec_layers.append(nn.Linear(self.dec_dim[-1], self.n_items))
        self.decoder = nn.Sequential(*dec_layers)
        self.decoder.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            self.truncated_normal_(m.bias, std=0.001)

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
        
    def forward(self, input_sup, input_que=None, weight=None):
        if self.training:
            h_context = self.encoder(input_sup)
            h_target = self.encoder(input_que)

            mu_context = h_context[:, :self.z_dim]
            logvar_context = h_context[:, self.z_dim:]

            mu_target = h_target[:, :self.z_dim]
            logvar_target = h_target[:, self.z_dim:]
            std_target = torch.exp(0.5 * logvar_target)

            eps = torch.randn_like(logvar_target)
            z_target = mu_target + eps * std_target

            logit = self.decoder(z_target)

            neg_ll, KL, loss = self.loss(mu_context, logvar_context, mu_target, logvar_target, input_que, logit, weight)
        else:
            h_context = self.encoder(input_sup)

            mu_context = h_context[:, :self.z_dim]
            logvar_context = h_context[:, self.z_dim:]
            std_context = torch.exp(0.5 * logvar_context)

            eps = torch.randn_like(logvar_context)
            z_context = mu_context + eps * std_context

            logit = self.decoder(z_context)

            neg_ll, KL, loss = None, None, None
        
        return logit, neg_ll, KL, loss

    def loss(self, mu_context, logvar_context, mu_target, logvar_target, input, logit, weight):
        log_softmax_logit = nn.functional.log_softmax(logit, dim=1)
        neg_ll = -torch.sum(weight * torch.sum(input * log_softmax_logit, dim=-1))

        KL = 0.5 * ((torch.exp(logvar_target) + (mu_target - mu_context) ** 2) / torch.exp(logvar_context) - 1. + (logvar_context - logvar_target))
        KL = torch.mean(torch.sum(KL, dim=-1))
        
        loss = neg_ll + KL
        return neg_ll, KL, loss


