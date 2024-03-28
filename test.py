# from ltp import LTP
# ltp = LTP("D:\model\small")
# segment, hidden = ltp.seg(["南京市长江大桥。"])
# print(segment)

from ltp import LTP
from ltp import StnSplit
ltp = LTP("D:\model\small")
 
sents = StnSplit().split("该僵尸网络包含至少35000个被破坏的Windows系统，攻击者和使用者正在秘密使用这些系统来开采Monero加密货币。该僵尸网络名为“ VictoryGate”，自2019年5月以来一直活跃。")
for i in sents:
    print(i,end="\n")
