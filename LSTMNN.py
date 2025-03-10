import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, LSTM, Sigmoid, ReLU, Dropout, Tanh, ModuleList, Softplus, ParameterList

from loss_function import integrateTrapezoid

"""
This snippet is to define the neural network model for the Hawkes process simulation.
"""

class CLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CLSTMCell, self).__init__()
        self.Wi = Linear(input_size + hidden_size, hidden_size)
        self.Wf = Linear(input_size + hidden_size, hidden_size)
        self.Wdelta = Linear(input_size + hidden_size, hidden_size)
        self.Wg = Linear(input_size + hidden_size, hidden_size)
        self.Wo = Linear(input_size + hidden_size, hidden_size)
        self.Sigmoid = Sigmoid()
        self.Tanh = Tanh()

    def forward(self, x, prev_hidden, prev_c):
        input_vec = torch.cat([x, prev_hidden], dim = -1)

        curr_i = self.Sigmoid(self.Wi(input_vec))
        curr_f = self.Sigmoid(self.Wf(input_vec))

        curr_g = torch.mul(curr_f, prev_c) + torch.mul(curr_i, self.Tanh(self.Wg(input_vec)))
        curr_delta = torch.exp(self.Wdelta(input_vec))

        curr_o = self.Sigmoid(self.Wo(input_vec))

        # curr_c = lambda t: curr_g + torch.mul((prev_c - curr_g), torch.exp(-curr_delta * t))
        # curr_hidden = lambda t: torch.mul(curr_o, self.Tanh(curr_c(t)))

        return curr_g, curr_o, curr_delta, prev_c - curr_g
    
class stackCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, event_type):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.event_type = event_type

        super(stackCLSTM, self).__init__()
        self.m_stack = ModuleList([CLSTMCell(input_size, hidden_size) for _ in range(event_type)])

        self.Wh = Linear((input_size + hidden_size) * event_type, hidden_size * event_type)
        self.m_alpha = ParameterList([nn.Parameter(torch.ones(hidden_size) / hidden_size) for _ in range(event_type)])

    def forward(self, x_time, dt_time):
        t_total, batch_size, _ = x_time.shape

        hidden_ini = [torch.zeros(batch_size, self.hidden_size, device=x_time.device) for _ in range(self.event_type)]
        c_ini = [torch.zeros(batch_size, self.hidden_size, device=x_time.device) for _ in range(self.event_type)]

        lambda_rtn = []
        lambda_integrate = []

        for t in range(t_total):
            x, dt = x_time[t], dt_time[t]
            # Concatenate input with all hidden states
            hidden_sum = torch.cat([torch.cat([x, hidden_ini[m]], dim=-1) for m in range(self.event_type)], dim=-1)
            hidden_sum = self.Wh(hidden_sum)
            
            curr_gs = []
            curr_os = []
            curr_deltas = []
            diffs = []

            # Process each event type
            for k in range(self.event_type):
                hidden_slice = hidden_sum[:, k * self.hidden_size: (k + 1) * self.hidden_size]
                curr_g, curr_o, curr_delta, diff = self.m_stack[k](x, hidden_slice, c_ini[k])
                curr_gs.append(curr_g)
                curr_os.append(curr_o)
                curr_deltas.append(curr_delta)
                diffs.append(diff)
            
            # Calculate intensities for each event type
            batch_lambdas = []
            batch_integrated_lambdas = []
            
            for j in range(self.event_type):
                # Calculate cell state at time dt
                c_t = curr_gs[j] + diffs[j] * torch.exp(-curr_deltas[j] * dt.unsqueeze(-1))
                # Calculate hidden state at time dt
                h_t = curr_os[j] * self.Tanh(c_t)
                
                # Calculate intensity using alpha weights
                intensity = torch.sum(self.m_alpha[j] * h_t, dim=1)
                batch_lambdas.append(intensity)
                
                # For integration, we need a function to evaluate at different times
                def intensity_func(t_val, batch_idx):
                    t_tensor = torch.tensor(t_val, device=x.device)
                    c_val = curr_gs[j][batch_idx] + diffs[j][batch_idx] * torch.exp(-curr_deltas[j][batch_idx] * t_tensor)
                    h_val = curr_os[j][batch_idx] * torch.tanh(c_val)
                    return torch.sum(self.m_alpha[j] * h_val).item()
                
                # Integrate for each batch element
                integrated_values = []
                for b in range(batch_size):
                    integrated = integrateTrapezoid(0, dt[b].item(), 100, lambda t: intensity_func(t, b))
                    integrated_values.append(integrated)
                
                batch_integrated_lambdas.append(torch.tensor(integrated_values, device=x.device))
            
            lambda_rtn.append(batch_lambdas)
            lambda_integrate.append(torch.stack(batch_integrated_lambdas).transpose(0, 1))
            
            # Update hidden and cell states for next timestep
            for k in range(self.event_type):
                c_ini[k] = curr_gs[k] + diffs[k] * torch.exp(-curr_deltas[k] * dt.unsqueeze(-1))
                hidden_ini[k] = curr_os[k] * self.Tanh(c_ini[k])

        return lambda_rtn, lambda_integrate