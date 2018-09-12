from django.shortcuts import render
import numpy as np
from sklearn.linear_model import LinearRegression
from backend.ECG_LSTM_DJANGO import *
from django.views.decorators import csrf
# Create your views here.

def search_post(request):
    ctx ={}
    if request.POST:
        ctx['rlt'] = request.POST['ml_csv']
    return render(request, "ml_post.html", ctx)

def lstm_ecg(request):
 acc = ""
 ctx = ""
 gnb_score = ""
 svc_score = ""
 knn_score = ""
 RF_score = ""
 if request.POST:
  ctx = request.POST['csv']
  df_cp = read_input(ctx)
  transformed_label = encode_label(df_cp)
  scaled = normalize(df_cp)
  model = models(transformed_label)
  gnb_score,svc_score,knn_score,RF_score = ml_train_score(df_cp)
  score,acc = train_score(scaled,transformed_label,model)
 context = {
		#'score':score,
		'gnb':gnb_score,
		'svc':svc_score,
		'knn':knn_score,
		'RF':RF_score,
		
		'ctx':ctx,
		'acc':acc,
   	}
 return render(request, 'lstm_ecg.html',context )