import torch
from TALENT.model.models.grande import GRANDE
from types import SimpleNamespace
import pickle

args = SimpleNamespace(
    cat_policy='indices',
)

import ipdb; ipdb.set_trace() 

trlog = torch.load("/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/LAMDA-TALENT/example_datasets/cpu_act-grande/Epoch5BZ1024-Norm-standard-Nan-mean-new-Cat-indices/trlog", map_location="cpu")
model_config = trlog['args']['config']['model']
batch_size = trlog['args']['batch_size']
task_type = 'regression'

model = GRANDE(
    batch_size=batch_size,
    task_type=task_type,
    **model_config
)

model.cat_idx = []
model.number_of_variables = 21
model.number_of_classes = 1
model.build_model()
checkpoint = torch.load("/pfs/work7/workspace/scratch/ma_fknuette-project_GRANDE/LAMDA-TALENT/example_datasets/cpu_act-grande/Epoch5BZ1024-Norm-standard-Nan-mean-new-Cat-indices/best-val-0.pth", map_location='cpu')

model.load_state_dict(checkpoint['params'])

model.eval()
model.double()

x = torch.randn(3, model.number_of_variables, dtype=torch.float64)
y = model(x)
print(y)