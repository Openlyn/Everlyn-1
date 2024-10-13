# pip install openxlab
import openxlab
openxlab.login(ak = 'orqw36pb4bqnoxn7l5yx', sk = '') 

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/GQA') 


from openxlab.dataset import get
get(dataset_repo='zzy8782180/HA-DPO', target_path='/')


from openxlab.dataset import download
download(dataset_repo='zzy8782180/HA-DPO',source_path='/README.md', target_path='') 