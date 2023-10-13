from matplotlib import pyplot as plt
from wrapper import Wrapper


number_of_observed_candles = 40

market = Wrapper(number_of_observed_candles)

all_days = market.get_available_days()

batch = market.get_states(all_days[125])

batch0 = batch[0]
candles1h_pair0 = batch0[1][0]
candles1h_pair1 = batch0[1][1]
candles1d_pair0 = batch0[1][2]
candles1d_pair1 = batch0[1][3]
candles1wk_pair0 = batch0[1][4]
candles1wk_pair1 = batch0[1][5]

plt.imshow(candles1d_pair0)
plt.tight_layout()
plt.show()

plt.imshow(candles1d_pair1)
plt.tight_layout()
plt.show()

plt.imshow(candles1wk_pair0)
plt.tight_layout()
plt.show()

plt.imshow(candles1wk_pair1)
plt.tight_layout()
plt.show()