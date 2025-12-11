# @title seq conv lcse
# ht = at*ht-1 + bt

def seq(at, bt, h0=None): # [t,b,d], [t,b,d], [b,d]
    h = 0 if h0==None else h0
    ht = []
    for t in range(at.size(0)):
        h = at[t] * h + bt[t]
        ht.append(h)
    ht = torch.stack(ht, dim=0)
    return ht

# conv
# h1=a1*h0+b1 =       a1*h0 +       b1
# h2=a2*h1+b2 =    a2*a1*h0 +    a2*b1 +    b2
# h3=a3*h2+b3 = a3*a2*a1*h0 + a3*a2*b1 + a3*b2 + b3

# h1/a1   = h0 + b1/a1
# h2/a12  = h0 + b1/a1 + b2/a12
# h3/a123 = h0 + b1/a1 + b2/a12 + b3/a123

def conv(at, bt, h0=None): # [t,b,d], [t,b,d], [b,d]
    a123 = torch.cumprod(at, dim=0) # [t,b,d]
    ht = torch.cumsum(bt/a123, dim=0) # cusum b1/a1, b2/a12, b3/a123, ...
    ht = ((h0 if h0!=None else 0)+ht)*a123
    return ht

# lcse
# computing the sequence ht = at*ht-1 + bt
# xt = e^(a*_t +tail(LCSE(cat(log x0, log b_t -a*_t))))
# b*_t = Sum t] e^( log bt - a*_t)
# a*_t = Sum t] log a_t
def lcse(at, bt, h0=None): # [t,b,d], [t,b,d], [b,d]
    if h0!=None:
        x_in = torch.cat([h0.log().unsqueeze(0), bt.log()], dim=0) # [1+t,b,d]
        at = torch.cat([torch.zeros(1,*at.shape[1:]), torch.cumsum(at.log(), dim=0)], dim=0) # a*_t # [1+t,b,d]
        ht = torch.exp(at[1:] + torch.logcumsumexp(x_in - at, dim=0)[1:])
    else:
        at = torch.cumsum(at.log(), dim=0) # a*_t # [t,b,d]
        ht = torch.exp(at + torch.logcumsumexp(bt.log() - at, dim=0))
    return ht


t, b, d = 30, 4, 5
# t, b, d = 200, 64, 64
# at = torch.randn(t,b,d, dtype=torch.complex64)
at = torch.randn(t,1,d, dtype=torch.complex64)
bt = torch.randn(t,b,d, dtype=torch.complex64)
h0 = torch.randn(b,d, dtype=torch.complex64)
h0 = None
ht = seq(at, bt, h0)
ht1 = conv(at, bt, h0)
# ht1 = lcse(at, bt, h0)
print(abs(ht-ht1).sum())
# print(ht[:3,0,:5])
# print(ht1[:3,0,:5])
