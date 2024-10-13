# pip install openxlab
import openxlab
openxlab.login(ak = '', sk = '') 
from openxlab.dataset import info
info(dataset_repo='OpenDataLab/GQA') 

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/GQA', target_path='') 

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/GQA',source_path='/README.md', target_path='') 