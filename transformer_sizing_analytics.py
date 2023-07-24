from agents.customer_agent import PowerCustomer
from agents.transformer_agent import Transformer
import pandas as pd
import matplotlib.pyplot as plt

start_date = '2008-01-01'
end_date = '2008-12-31'

transformer_sizing = PowerCustomer(unique_id=None,
                                   model=None,
                                   yearly_cons_household=3500,
                                   start_date=start_date,
                                   end_date=end_date)

transformer_sizing.initialize_customer()
peak_load = transformer_sizing.get_peak_load_kw()

sizing = {}
list = [25, 50, 150, 300]
# list = [x for x in range(401)]

for i in list:
    transformer = Transformer(unique_id=1,
                              model=None,
                              num_households=i,
                              peak_load=peak_load)
    capacity = transformer.get_capacity_kw()
    print(capacity)
    all_peak = peak_load * i
    sizing[i] = (capacity, all_peak)

# df = pd.DataFrame(sizing.items(), columns=['num_customers', 'transformer_capacity'])
df = pd.DataFrame.from_dict(sizing, orient='index', columns=['capacity', 'peak_load'])
intersection_index = df[df['peak_load'] >= df['capacity']].index.min()
df.plot(color=['dimgrey', 'black'])
plt.fill_between(df.index, df['capacity'], df['peak_load'], hatch='//////', color='lightgray', edgecolor='black', facecolor='none', where=df.index <= intersection_index, alpha=0.3, label='p_over')
plt.xlim(0, None)
plt.ylim(0, None)
plt.title('Transformer sizing')
plt.ylabel('Transformer capacity \n [kW]')
plt.xlabel('Number of customers')
plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.2), frameon=False)
plt.tight_layout()
fig_name = 'transformer_sizing'
# plt.savefig(fig_name, dpi=300)
plt.show()