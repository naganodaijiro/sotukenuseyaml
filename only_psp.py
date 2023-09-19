import time#処理時間を図る
import os#ファイルがない場合に作るため
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image,ImageDraw,ImageFilter
from argparse import Namespace
#alignmentに必要
#from pixel2style2pixel.scripts.align_all_parallel import align_face
#from pixel2style2pixel.scripts.align_all_parallel import get_landmark
import torch
import torchvision.transforms as transforms
from pixel2style2pixel.utils import common
from pixel2style2pixel.models.psp import pSp

#sessionの作成
if 'size' not in st.session_state:
  st.session_state.size = None
if 'ckpt' not in st.session_state:
  st.session_state.ckpt = None
if 'net' not in st.session_state:
  st.session_state.net = None
if 'device' not in st.session_state:
  st.session_state.device = None
if 'old_csv' not in st.session_state:
  st.session_state.old_csv = None
if 'predictor' not in st.session_state:
   st.session_state.predictor = None
if 'detector' not in st.session_state:
   st.session_state.detector = None


experiment_type = 'ffhq_encode'
# モデルの設定
EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pretrained_models/psp_ffhq_encode.pt",
        #"image_path": path1,
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    } 
}




EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

if st.session_state.size is not None:
  print("chash済み"+str(st.session_state.size))
else:
  st.session_state.size = os.path.getsize(EXPERIMENT_ARGS['model_path'])
  print(str(st.session_state.size))
  if st.session_state.size < 1000000:
    raise ValueError("Pretrained model was unable to be downlaoded correctly!")

model_path = EXPERIMENT_ARGS['model_path']

if st.session_state.ckpt is None:
  st.session_state.ckpt = torch.load(model_path, map_location='cpu')
ckpt = st.session_state.ckpt

opts = ckpt['opts']
#pprint.pprint(opts)
if st.session_state.net is None:
  # update the training options
  opts['checkpoint_path'] = model_path
  if 'learn_in_w' not in opts:
      opts['learn_in_w'] = False
  if 'output_size' not in opts:
      opts['output_size'] = 1024
  opts = Namespace(**opts)
  st.session_state.device = opts.device 
  st.session_state.net = pSp(opts)

net = st.session_state.net
net.eval()
net.cuda()



#エンコーダ用,image_to_latent
def image_to_latent(img_path):
  input_image = Image.open(img_path)
  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image)

  w=net.encoder(transformed_image.unsqueeze(0).to("cuda").float())
  w=w+ckpt['latent_avg'].to(st.session_state.device).repeat(w.shape[0],1,1)
  return w

#デコーダ用、image_to_latent2,w_to_tensor,re_const
def image_to_latent2(img_path):
  input_image = Image.open(img_path)
  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image)

  w=net.encoder(transformed_image.unsqueeze(0).to("cuda").float())
  #w=w+ckpt['latent_avg'].to(opts.device).repeat(w.shape[0],1,1)
  return w

def w_to_tensor(w):
  decode_tensor,result_latent=net.decoder([w],input_is_latent=True,randomize_noise=False,return_latents=True)
  decode_tensor=net.face_pool(decode_tensor)[0]
  return decode_tensor

def re_const(coef2,concept_vector,bak_w):
  smile_vector = concept_vector * coef2

  with torch.no_grad():
    ww = bak_w
    ww=ww+ckpt['latent_avg'].to(st.session_state.device).repeat(ww.shape[0],1,1)

    ww[0]=ww[0]+smile_vector

    result_tensor2 = w_to_tensor(ww)
    result_img2 = common.tensor2im(result_tensor2)
  return result_img2



pagelist = ["デコーダ"]
#st.set_page_config(layout="wide")
st.title('表情変化の検証用システム')
st.markdown("エンコード、デコード、ベクトルの計算を行うことができます")
selector=st.sidebar.radio( "ページ選択",pagelist)



if selector=="デコーダ":
  
    st.title("デコーダ")
    st.markdown("# csvファイルと画像をアップロードしてください")
    uploaded_file = st.file_uploader("Choose an csv...",type="csv")
    uploaded_file_image = st.file_uploader("Choose an image...",type="jpg")
    if uploaded_file is not None:
        csv_path = uploaded_file
        uv1 = pd.read_csv(csv_path, index_col=0)
        uv2 = uv1.to_numpy()
        uv3 = torch.from_numpy(uv2)
        concept_vector = uv3.to('cuda')

        vn = np.linalg.norm(uv2)
        vn_max = np.amax(uv2,axis=0)



        
        if (uploaded_file_image is not None) and (uploaded_file is not None):
            IMG_PATH = uploaded_file_image
            img = Image.open(IMG_PATH)
            imgr = np.array(img)
            col1,col2 = st.columns(2)
            level = st.slider('how level you want to change',  min_value=1, max_value=100, step=1, value=100)
            level = level / 100
            with col1:
              st.header("オリジナル画像")
              st.image(imgr,use_column_width = True)

        
            with torch.no_grad():
              input_w = image_to_latent2(IMG_PATH)
            bak_w =input_w
            time_sta = time.time()
            imim = re_const(level,concept_vector,bak_w)
            time_end = time.time()
            tim = time_end-time_sta
            
            # 結果の出力
            with col2:
              st.header("変化後画像")
              st.image(imim,use_column_width=True)
            st.markdown("## time:" + str(tim))


