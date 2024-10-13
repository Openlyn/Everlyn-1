# pip install openxlab
import openxlab
openxlab.login(ak = '9dyve1ng8dxlb736bdan', sk = 'ebny1kk2z37aqmvpype2rz50oga6ejdgymwxon4z') 
from openxlab.dataset import info
info(dataset_repo='OpenDataLab/Visual_Genome_Dataset_V1_dot_2')

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/Visual_Genome_Dataset_V1_dot_2', target_path='') 

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/Visual_Genome_Dataset_V1_dot_2',source_path='/README.md', target_path='') 