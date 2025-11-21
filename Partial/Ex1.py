from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


model = DiscreteBayesianNetwork([('O', 'H'), ('O', 'W'), ('H', 'R'), ('W', 'R'), ('H', 'E'), ('R', 'C')])

cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.7], 
                           [0.3]])

cpd_h = TabularCPD(variable='H', variable_card=2,
                   values=[[0.1, 0.8],  
                           [0.9, 0.2]], 
                   evidence=['O'],
                   evidence_card=[2])
cpd_w = TabularCPD(variable='W', variable_card=2,
                   values=[[0.9, 0.4],  
                           [0.1, 0.6]], 
                   evidence=['O'],
                   evidence_card=[2])
cpd_r = TabularCPD(variable='R', variable_card=2,
                   values=[[0.5, 0.7, 0.1, 0.4],  
                           [0.5, 0.3, 0.9, 0.6]], 
                   evidence=['H', 'W'],
                   evidence_card=[2, 2])
cpd_e = TabularCPD(variable='E', variable_card=2,
                   values=[[0.2, 0.8],  
                           [0.8, 0.2]], 
                   evidence=['H'],
                   evidence_card=[2])
cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.15, 0.4],  
                           [0.85, 0.6]], 
                   evidence=['R'],
                   evidence_card=[2])

model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)
inference = VariableElimination(model)
prob_h_given_c = inference.query(variables=['H'], evidence={'C': 1})
print(prob_h_given_c)

prob_e_given_c = inference.query(variables=['E'], evidence={'C': 1})
print(prob_e_given_c)

map_estimate = inference.map_query(variables=['H', 'W'], evidence={'C': 1})
print(map_estimate)


pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

independencies = model.get_independencies()
print(independencies)