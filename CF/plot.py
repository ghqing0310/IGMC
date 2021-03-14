from pyecharts import options as opts
from pyecharts.charts import Graph
import numpy as np
import pandas as pd
import networkx as nx


users_m = np.load('users_100k.npy')
users_m[range(users_m.shape[0]), range(users_m.shape[0])] = 0
link_m = users_m.argsort()[:, -100:][:, ::-1]
link1, link2 = np.where(users_m > 0.0379)

edge_index = list(zip(link1, link2))

user_df = pd.read_csv(
    'u.user', names=['user_id', 'age', 'sex', 'occupation', 'zip_code'], header=None, sep='|')

subnode = list(set(link1.tolist()) | set(link2.tolist()))

G = nx.Graph()
G.add_nodes_from(subnode)
G.add_edges_from(edge_index)

bet_cen = nx.betweenness_centrality(G)

nodes = [{'name': str(i), 'category': user_df.iloc[i]
          ['occupation']} for i in subnode]
print(len(nodes))

links = [{'source': str(e[0]), 'target': str(e[1])} for e in zip(link1, link2)]
print(len(links))

categories = [{'name': oc} for oc in list(set(user_df['occupation'].tolist()))]

c = (
    Graph(init_opts=opts.InitOpts(bg_color='rgba(255,250,205,0.2)',
                                  width='1500px',
                                  height='700px',
                                  page_title='page'))
    .add("", nodes, links, categories=categories, repulsion=4000, label_opts=opts.LabelOpts(is_show=False))
    .render("user.html")
)
