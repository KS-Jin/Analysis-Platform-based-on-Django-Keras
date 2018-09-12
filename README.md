# 鄉民來幫忙之運動競賽比分預測系統
為中興大學-社群網路與運算 分組期末專案  
題目要求取得社群的資料,並從中構思  

## 簡介
由於我們熱愛運動賽事而發想了這樣的主題  
選擇了在台灣較熱門的運動-籃球 為我們資料收集的來源  
從實際的大比分變化來觀察鄉民的推文是否真實反映比賽狀況    

## 資料爬蟲與整理格式
[west_series_crawl.ipynb](https://github.com/KS-Jin/NBA-prediction-by-PTT-retweets/blob/master/west_series_crawl.ipynb)  
由於鄉民對於球員有需多稱呼與綽號,建立的綽號詞袋收集  
也建立了貶抑詞詞袋做為反向指標  
隊伍名稱/教練/先發五名球員/替補九名成員  
基本欄位共16個，判對好壞後共32個特徵  

## 預測比分勝負模型
[nba_two_team.ipynb](https://github.com/KS-Jin/NBA-prediction-by-PTT-retweets/blob/master/nba_two_team.ipynb)  
分別對於冠軍賽的兩對收集它們季後賽的推文資料作為訓練資料  
預測冠軍賽的真實得分情況與勝負
  
## 分析結果1
![image](https://github.com/KS-Jin/NBA-prediction-by-PTT-retweets/blob/master/pic/pic1.JPG) 
 顯示我們使用鄉民推文作為特徵來預測NBA賽事的勝負有 100% 準確率  
 而在得分方面則有些許誤差  
 
## 分析結果2
![image](https://github.com/KS-Jin/NBA-prediction-by-PTT-retweets/blob/master/pic/pic2.JPG)  
使用這樣的結果在國際賭盤上來預測大小讓分  
比起單純預測得分比還更加準確  

## 其他資料
火箭隊季後賽推文情況 [rockets_west.csv](https://github.com/KS-Jin/NBA-prediction-by-PTT-retweets/blob/master/rockets_west.csv)  
勇士隊季後賽推文情況 [warriors_west.csv](https://github.com/KS-Jin/NBA-prediction-by-PTT-retweets/blob/master/warriors_west.csv)  
報告文檔 [鄉民來幫忙之運動競賽比分預測系統.pdf](https://github.com/KS-Jin/NBA-prediction-by-PTT-retweets/blob/master/%E9%84%89%E6%B0%91%E4%BE%86%E5%B9%AB%E5%BF%99%E4%B9%8B%E9%81%8B%E5%8B%95%E7%AB%B6%E8%B3%BD%E6%AF%94%E5%88%86%E9%A0%90%E6%B8%AC%E7%B3%BB%E7%B5%B1.pdf)  


 