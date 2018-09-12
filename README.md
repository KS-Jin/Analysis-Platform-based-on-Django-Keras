# Analysis Platform based on Django & Keras
AWIN Lab 多維生醫感測資訊計畫之平台雛形  
作為專家系統的 Server端,預計下個階段開發手持裝置與此系統串接  

## 簡介
使用 Django 做為後端Server 方便與 Python資料分析的模型對接     
使用了 LSTM的模型與傳統的分類器15種病症辨識的準確率比較    
透過簡單的輸入選擇病患編號,來分析個分類器對於此題目的準確性  

## 訓練病症分辨模型
[ECG_LSTM_DJANGO.py](https://github.com/KS-Jin/Analysis-Platform-based-on-Django-Keras/blob/master/backend/ECG_LSTM_DJANGO.py)  
為後端處理資料之主程式,資料之前處理、標準化、encoding、模型訓練    

## 前端處理
[lstm_ecg.html](https://github.com/KS-Jin/Analysis-Platform-based-on-Django-Keras/blob/master/templates/lstm_ecg.html)    
採用 Vue.js 接收辨識評比的分數  
  
## 呈現畫面
![image](https://github.com/KS-Jin/Analysis-Platform-based-on-Django-Keras/blob/master/pic/pic1.jpg)   
列出各模型對於辨識的準確率  
 
## 分類器比較
![image](https://github.com/KS-Jin/Analysis-Platform-based-on-Django-Keras/blob/master/pic/pic2.jpg)  
可以看出使用 LSTM模型在此問題的效果比傳統分類器還要精確  

## 其他資料
MIT-BIH 病患資料集 [MITBIH_CSV](https://github.com/KS-Jin/Analysis-Platform-based-on-Django-Keras/tree/master/MITBIH_CSV)    
報告文檔 [Analysis Platform based on Django & Keras.pdf](https://github.com/KS-Jin/Analysis-Platform-based-on-Django-Keras/blob/master/Analysis%20Platform%20based%20on%20Django%20%26%20Keras.pdf)  


 