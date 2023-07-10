# Enhancing Product Discovery in Exhibition 📈

<!-- <font size=+3><center><b>Telco Churn Prediction with ML Insights 📈</b></center></font> -->
<div style="text-align: center;">
  <img src="rec.png" alt="Image" style="display: block; margin: 0 auto;" width="360" height="360" />
</div>
<span style="font-size: 12px;"><center><em>Photo by NVIDIA</em></center></span> <br>
<span style="font-size: 20px;"><left><b>Table of Contents</b></left></span>

- [Introduction](#Introduction)
- [Objective](#Objective)
- [Libraries](#Libraries)
- [Default Setting](#Default-Setting)
- [Functions](#Functions)
- [A Quick Look at our Data](#A-Quick-Look-at-our-Data)
    - [Data Attributes](#Dataset-Attributes)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
    - [Continuous Variables](#Continuous-Variables)
    - [Categorical Variables](#Categorical-Variables)
- [Data Preprocessing](#Data-Preprocessing)
    - [Encoding Categorical Features](#Encoding-Categorical-Features)
    - [Text Encoding](#Text-Encoding)
- [Recommendation Model Building](#Recommendation-Model-Building)
    - [Experiment and Parameter Tuning](#Experiment-and-Parameter-Tuning)
    - [Recommend Top K](#Recommend-Top-K)
- [Future Development](#Future-Development)
- [Conclusions](#Conclusions)

# Introduciton

In the dynamic exhibition industry, attendees often face the challenge of efficiently discovering relevant products amidst a vast array of offerings. To address this issue, recommendation systems have gained significant attention for their ability to provide personalized suggestions based on user preferences. In this research, we aim to develop an item-based recommendation model tailored to the exhibition industry, leveraging the text descriptions of products available. By analyzing these features, our model will enable attendees to find products that closely align with their interests and recommend similar items, thereby enhancing the overall exhibition experience.

# Objective

In this case, the absence of explicit labels and user information poses a challenge for building effective recommendation models. This research aims to develop an item-based recommendation model using product features, leveraging unsupervised learning techniques to overcome the absence of labels and limited user information. By addressing this problem, the study aims to enhance product discovery and facilitate meaningful connections between attendees and exhibitors in the exhibition industry.

In this project, I would like to answer intriguing questions that I have discovered:

* What are the key product features that significantly influence the relevance and similarity of items within the exhibition industry? How can the item-based recommendation model effectively utilize these product features?
* What are the suitable similarity analysis techniques and algorithms that can measure item similarity based on the available product features?
* How can machine learning techniques be employed to train the item-based recommendation model using the exhibition industry's product feature dataset?
* What are the challenges and techniques involved in extracting relevant information from the text descriptions of products to enhance the recommendation model?
* How can the performance and accuracy of the recommendation model be evaluated and measured in the context of the exhibition industry, considering the absence of explicit labels?

# Libraries


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Write/Read Excel 
import openpyxl

import numpy as np
import pandas as pd
pd.set_option('precision', 3)

# Data Visualisation Libraries
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

!pip install seaborn --upgrade
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

# Sentence BERT
from sentence_transformers import SentenceTransformer, util
print('✔️ Libraries Imported!')
```

    Requirement already satisfied: seaborn in c:\users\user\anaconda3\lib\site-packages (0.12.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.17 in c:\users\user\anaconda3\lib\site-packages (from seaborn) (1.20.1)
    Requirement already satisfied: pandas>=0.25 in c:\users\user\anaconda3\lib\site-packages (from seaborn) (1.2.4)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in c:\users\user\anaconda3\lib\site-packages (from seaborn) (3.3.4)
    Requirement already satisfied: cycler>=0.10 in c:\users\user\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\user\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.4.7)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\user\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (8.2.0)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.8.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.3.1)
    Requirement already satisfied: six in c:\users\user\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib!=3.6.1,>=3.1->seaborn) (1.15.0)
    Requirement already satisfied: pytz>=2017.3 in c:\users\user\anaconda3\lib\site-packages (from pandas>=0.25->seaborn) (2021.1)
    ✔️ Libraries Imported!
    

# Default Setting


```python
pd.options.display.max_rows = None
pd.options.display.max_columns = None

font_size = 18
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size + 2
plt.rcParams['xtick.labelsize'] = font_size - 2
plt.rcParams['ytick.labelsize'] = font_size - 2
plt.rcParams['legend.fontsize'] = font_size - 2

# colors = ['#00A5E0', '#DD403A']
colors_cat = ['#E8907E', '#D5CABD', '#7A6F86', '#C34A36', '#B0A8B9', '#845EC2', '#8f9aaa', '#FFB86F', '#63BAAA', '#9D88B3', '#38c4e3']
# colors_comp = ['steelblue', 'seagreen', 'black', 'darkorange', 'purple', 'firebrick', 'slategrey']

random_state = 42
# scoring_metric = 'recall'
# comparison_dict, comparison_test_dict = {}, {}

print('✔️ Default Setting Done!')
```

    ✔️ Default Setting Done!
    

# Functions

## normalize_embedding()


```python
def normalize_embedding(embedding, norm=2):
    '''
    Normalize the input embedding vector.

    Args:
        embedding (numpy.ndarray): 
            The input embedding vector to be normalized.
        
        norm (int, default=2): 
            The order of the norm to be applied for normalization.
        
    Returns:
        numpy.ndarray:
            The normalized embedding vector.
    '''
    embedding = embedding / np.linalg.norm(embedding, ord=norm)
    return embedding
```

## recommend_top_k(product_id, k, threshold)


```python
def recommend_top_k(product_id: int, k: int, threshold: float) -> tuple:
    '''
    Recommend the top k similar products given a product ID.

    Args:
        product_id (int): 
            The ID of the product for which recommendations will be generated.

        k (int): 
            The number of top similar products to recommend.

        threshold (float): 
            The threshold value for similarity scores. 
            Only recommendations with scores equal to or greater than this threshold will be considered.

    Returns:
        tuple:
            A tuple containing two lists:
            - The recommended product IDs.
            - The corresponding similarity scores.
            - The row indices of the recommended products as a list.

    Example:
        rec_pd_ids, rec_scores, rec_row_indices = recommend_top_k(CU0004601801, 5)
    '''
    

    row_index = df[df['Product_id'] == product_id].index
    # the reason for using k+1 is because the given item will be recommended
    rec_k_dic = util.semantic_search(embedding[row_index], embedding, top_k=k+1)[0] 
    # drop the given item itself
    rec_k_dic = np.delete(rec_k_dic, 0)
    rec_row_idx = []
    rec_score_ls = []
    for item in rec_k_dic:
        score = round(item['score'], 3)
        if score >= threshold:
            rec_row_idx.append(item['corpus_id'])
            rec_score_ls.append(round(item['score'], 3)) 

    rec_pd_id_ls = np.array(df.loc[rec_row_idx, 'Product_id'])
    return (rec_pd_id_ls, rec_score_ls, rec_row_idx)
```

# A Quick Look at our Data

## Dataset Attributes

- **Product_id**: The unique ID of the product.
- **Product_Name**: The name or title of the product in Chinese.
- **Vendor_id**: The unique ID of the vendor.
- **Main_Category**: The main category or exhibition type of the product.
- **Sub_Category**: The sub-category or specific category of the product.
- **Description**: The text description of the product in Chinese.
- **Product_Name_en**: The name or title of the product in English.
- **Description_en**: The text description of the product in English.


```python
df = pd.read_excel('./Data/Product_20230702.xlsx', header=0, skiprows=[0,2,3,4])

print('✔️ Dataset Imported Successfully!\n')
print('It contains {} rows and {} columns.'.format(df.shape[0], df.shape[1]))
```

    ✔️ Dataset Imported Successfully!
    
    It contains 977 rows and 28 columns.
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>語系*</th>
      <th>產品ID*</th>
      <th>產品名稱*</th>
      <th>自定關鍵字(逗點隔開)</th>
      <th>是否要貼標</th>
      <th>庫存狀況</th>
      <th>定價</th>
      <th>售價</th>
      <th>產品網址*</th>
      <th>廠商ID*</th>
      <th>廠商攤位號碼</th>
      <th>廠商名稱*</th>
      <th>國家簡寫*</th>
      <th>原廠名稱</th>
      <th>產品主類別*(展別)</th>
      <th>產品次類別*(類別)</th>
      <th>簡述 (展品特色)</th>
      <th>規格</th>
      <th>SEO Title</th>
      <th>SEO DES</th>
      <th>影片連結</th>
      <th>認證</th>
      <th>是否開啟*</th>
      <th>列表圖</th>
      <th>內頁圖片1</th>
      <th>Unnamed: 25</th>
      <th>Unnamed: 26</th>
      <th>Unnamed: 27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tw</td>
      <td>CU0004601801</td>
      <td>PE 修補膠帶</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>in_stock</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>CU0004601801</td>
      <td>CU00046018</td>
      <td>NaN</td>
      <td>萬洲化學股份有限公司</td>
      <td>TW</td>
      <td>萬洲化學股份有限公司</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>●全天候環保聚乙烯膠帶\n \n ●高粘合溶劑型丙烯酸粘合劑 \n \n ●適用於大範圍的戶...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>CU0004601801.jpg</td>
      <td>CU0004601801.jpg</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tw</td>
      <td>CU0004601802</td>
      <td>C+ 生物可分解膠帶</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>in_stock</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>CU0004601802</td>
      <td>CU00046018</td>
      <td>NaN</td>
      <td>萬洲化學股份有限公司</td>
      <td>TW</td>
      <td>萬洲化學股份有限公司</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>● C+通用型包裝膠帶\n \n ●全世界第一個生物可分解OPP 包裝解決方案，可與一般PP...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>CU0004601802.jpg</td>
      <td>CU0004601802.jpg</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tw</td>
      <td>CU0004601803</td>
      <td>PVC 接梨膠帶</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>in_stock</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>CU0004601803</td>
      <td>CU00046018</td>
      <td>NaN</td>
      <td>萬洲化學股份有限公司</td>
      <td>TW</td>
      <td>萬洲化學股份有限公司</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>●軟質亮面PVC 膠帶\n \n ●高剝離力\n \n ●適合用於梨子接枝 \n \n ●環...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>CU0004601803.jpg</td>
      <td>CU0004601803.jpg</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tw</td>
      <td>CU0004601804</td>
      <td>PVC 接梨膠帶</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>in_stock</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>CU0004601804</td>
      <td>CU00046018</td>
      <td>NaN</td>
      <td>萬洲化學股份有限公司</td>
      <td>TW</td>
      <td>萬洲化學股份有限公司</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>軟質亮面PVC 膠帶、高剝離力、適合用於梨子接枝</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>CU0004601804.jpg</td>
      <td>CU0004601804.jpg</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tw</td>
      <td>CU0004601805</td>
      <td>回收PET膠帶</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>in_stock</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>CU0004601805</td>
      <td>CU00046018</td>
      <td>NaN</td>
      <td>萬洲化學股份有限公司</td>
      <td>TW</td>
      <td>萬洲化學股份有限公司</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>高黏著力、無溶劑、環境友善、高保持力、高機械強度</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>CU0004601805.jpg</td>
      <td>CU0004601805.jpg</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Remove redundant variables


```python
df = df[['語系*', '產品ID*', '產品名稱*', '廠商ID*', '國家簡寫*', '產品主類別*(展別) ', '產品次類別*(類別)', '簡述 (展品特色)']]
```


```python
translation_dict = {
    '語系*': 'Language',
    '產品ID*': 'Product_id',
    '產品名稱*':'Product_Name',
    '廠商ID*': 'Vendor_id',
    '國家簡寫*': 'Country',
    '產品主類別*(展別) ': 'Main_Category',
    '產品次類別*(類別)': 'Sub_Category',
    '簡述 (展品特色)': 'Description'
}
df = df.rename(columns=translation_dict)
```

To combine descriptions in different languages, perform a self-join on the table.


```python
df_tw = df[df['Language']=='tw'].reset_index(drop = True, inplace = False)
df_en = df[df['Language']=='en'].reset_index(drop = True, inplace = False)
```


```python
df = pd.merge(df_tw, df_en, on='Product_id', how='outer')
```


```python
col_idx = np.concatenate((np.arange(1,8,1) , [df.columns.get_loc('Product_Name_y'),-1]))
df = df.iloc[:,col_idx]
```


```python
df.columns
```




    Index(['Product_id', 'Product_Name_x', 'Vendor_id_x', 'Country_x',
           'Main_Category_x', 'Sub_Category_x', 'Description_x', 'Product_Name_y',
           'Description_y'],
          dtype='object')




```python
col_name = ['Product_id', 'Product_Name', 'Vendor_id', 'Country', 'Main_Category',
            'Sub_Category', 'Description','Product_Name_en', 'Description_en']
df.columns = col_name
```

The `info()` method can give us valuable information such as the number of non-null values and the type of each feature:


```python
df.info()
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 488 entries, 0 to 487
    Data columns (total 9 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Product_id       488 non-null    object
     1   Product_Name     488 non-null    object
     2   Vendor_id        488 non-null    object
     3   Country          488 non-null    object
     4   Main_Category    488 non-null    object
     5   Sub_Category     488 non-null    object
     6   Description      487 non-null    object
     7   Product_Name_en  488 non-null    object
     8   Description_en   487 non-null    object
    dtypes: object(9)
    memory usage: 38.1+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_id</th>
      <th>Product_Name</th>
      <th>Vendor_id</th>
      <th>Country</th>
      <th>Main_Category</th>
      <th>Sub_Category</th>
      <th>Description</th>
      <th>Product_Name_en</th>
      <th>Description_en</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CU0004601801</td>
      <td>PE 修補膠帶</td>
      <td>CU00046018</td>
      <td>TW</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>●全天候環保聚乙烯膠帶\n \n ●高粘合溶劑型丙烯酸粘合劑 \n \n ●適用於大範圍的戶...</td>
      <td>PE repair tape</td>
      <td>●All-weather environmentally friendly polyethy...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CU0004601802</td>
      <td>C+ 生物可分解膠帶</td>
      <td>CU00046018</td>
      <td>TW</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>● C+通用型包裝膠帶\n \n ●全世界第一個生物可分解OPP 包裝解決方案，可與一般PP...</td>
      <td>C biodegradable tape</td>
      <td>● C+ general purpose packing tape\n \n ●The wo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CU0004601803</td>
      <td>PVC 接梨膠帶</td>
      <td>CU00046018</td>
      <td>TW</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>●軟質亮面PVC 膠帶\n \n ●高剝離力\n \n ●適合用於梨子接枝 \n \n ●環...</td>
      <td>PVC pear tape</td>
      <td>●Soft and glossy finished PVC tape\n \n ●High ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CU0004601804</td>
      <td>PVC 接梨膠帶</td>
      <td>CU00046018</td>
      <td>TW</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>軟質亮面PVC 膠帶、高剝離力、適合用於梨子接枝</td>
      <td>PVC pear tape</td>
      <td>Soft and glossy finished PVC tape、High unwindi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CU0004601805</td>
      <td>回收PET膠帶</td>
      <td>CU00046018</td>
      <td>TW</td>
      <td>agritech</td>
      <td>Garden-Materials</td>
      <td>高黏著力、無溶劑、環境友善、高保持力、高機械強度</td>
      <td>Recycled PET Tape</td>
      <td>Strong Adhesion、Solvent-Free、ECO-Friendly、Heav...</td>
    </tr>
  </tbody>
</table>
</div>



Modify the data type of category variables


```python
categorical = ['Country', 'Main_Category', 'Sub_Category']
df[categorical] = df[categorical].astype('category')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 488 entries, 0 to 487
    Data columns (total 9 columns):
     #   Column           Non-Null Count  Dtype   
    ---  ------           --------------  -----   
     0   Product_id       488 non-null    object  
     1   Product_Name     488 non-null    object  
     2   Vendor_id        488 non-null    object  
     3   Country          488 non-null    category
     4   Main_Category    488 non-null    category
     5   Sub_Category     488 non-null    category
     6   Description      487 non-null    object  
     7   Product_Name_en  488 non-null    object  
     8   Description_en   487 non-null    object  
    dtypes: category(3), object(6)
    memory usage: 29.7+ KB
    


```python
df['Main_Category'] = df['Main_Category'].replace({0:'N/A'})
df['Sub_Category'] = df['Sub_Category'].replace({0:'N/A'})
```

# Exploratory Data Analysis


```python
df['Vendor_id'].value_counts().hist(figsize=(8, 5),
                          bins=20,
                          color='steelblue',
                          linewidth=1.5);
plt.title('Number of Products per Vendor')
plt.xlabel('Number of Products')
plt.ylabel('Counts')
```




    Text(0, 0.5, 'Counts')




    
![png](README_files/README_32_1.png)
    



```python
df['Country'].value_counts()
```




    TW    487
    IT      1
    Name: Country, dtype: int64




```python
df.drop(columns = 'Country', inplace = True)
```

## Categorical variable


```python
df['Main_Category'].value_counts()
```




    agritech                        280
    agrilivestock                   116
    agrifresh                        53
    sustainable-aquatic-products     25
    fish-farming                     14
    Name: Main_Category, dtype: int64




```python
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Main_Category', data=df, palette=colors_cat, width=0.8)
plt.ylabel('Counts')
plt.xticks(rotation=25)
plt.tight_layout();
```


    
![png](README_files/README_37_0.png)
    


For certain subgroups, there are only a few products.


```python
df['Sub_Category'].value_counts()
```




    Agritech-Other                                           129
    Agrilivestock-Other                                       50
    Labour-Saving-Machinery-and-Equipment                     40
    Livestock-Feed-and-Additives                              37
    Garden-Materials                                          29
    Sustainable-Aquatic-Products-Other                        24
    Intelligent-Detection-System-and-Equipment                23
    Ventilation-Equipment                                     21
    Organic-Fertilizer                                        15
    Agrifresh-Other                                           13
    Agricultural-Processing-Machinery                         12
    Seedlings-and-Flower-Seed                                 11
    Plant-Disease-and-Pest-Control                            11
    AIoT-Intelligent-Cold-Chain-Logistic-Solution             10
    Refrigeration-and-Freezing-Equipment                      10
    Fish-Farming-Other                                         9
    Intelligent-Temperature-Control-Technology                 8
    AIoT-Equipment-and-System                                  8
    Intelligent-Environmental-Control-Devices-and-Systems      5
    Grow-Light                                                 4
    Agricultural-Automation-Equipment                          4
    Intelligent-Irrigation-System-and-Equipment                3
    Water-Quality-Improve                                      3
    Biotechnology-Applications                                 2
    Feed-and-Feed-Additive                                     1
    Feed-Processing-Equipment-and-Testing-Equipment            1
    Organic-Waste-Disposal-Technology-and-Equipment            1
    Dehydrated-and-Pickled-Aquatic-Products                    1
    Artificial-Fog-Equipment                                   1
    Aquaculture-Technology-and-Management                      1
    Thermal-Camera                                             1
    Name: Sub_Category, dtype: int64



## Text Description


```python
df['Description'].head()
```




    0    ●全天候環保聚乙烯膠帶\n \n ●高粘合溶劑型丙烯酸粘合劑 \n \n ●適用於大範圍的戶...
    1    ● C+通用型包裝膠帶\n \n ●全世界第一個生物可分解OPP 包裝解決方案，可與一般PP...
    2    ●軟質亮面PVC 膠帶\n \n ●高剝離力\n \n ●適合用於梨子接枝 \n \n ●環...
    3                             軟質亮面PVC 膠帶、高剝離力、適合用於梨子接枝
    4                             高黏著力、無溶劑、環境友善、高保持力、高機械強度
    Name: Description, dtype: object




```python
df['Description_en'].head()
```




    0    ●All-weather environmentally friendly polyethy...
    1    ● C+ general purpose packing tape\n \n ●The wo...
    2    ●Soft and glossy finished PVC tape\n \n ●High ...
    3    Soft and glossy finished PVC tape、High unwindi...
    4    Strong Adhesion、Solvent-Free、ECO-Friendly、Heav...
    Name: Description_en, dtype: object



# Data Preprocessing

We will create four embeddings:

* Main Category embedding 
* Subcategory embedding 
* Product Name embedding
* Product Description embedding

To generate accurate recommendation, we determine the weights of these embeddings based on their importance.


```python
arr = [1, 4, 25, 13]
alpha = arr/np.sum(arr)
```

## Categorical Features Encoding

Utilize one-hot encoding to encode the `Main_Category` and `Sub_Category` variables, enabling us to analyze the similarity between vectors in the further analysis


```python
main_encode = pd.get_dummies(df['Main_Category'], drop_first=False)*alpha[0]
sub_encode = pd.get_dummies(df['Sub_Category'], drop_first=False)*alpha[1]
category_encode = np.concatenate((main_encode, sub_encode), axis=1)
category_encode.shape
```




    (488, 36)



## Text Encoding

Utilize Sentence-BERT, which is based on Siamese BERT-Networks, to generate embeddings for both `Product_Name` and `Description`, as described in the paper ["Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://arxiv.org/pdf/1908.10084.pdf) (Reimers & Gurevych, 2019).


```python
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
product_em_tw_ = model.encode(df['Product_Name'])
product_em_en_ = model.encode(df['Product_Name_en'])
des_em_tw_ = model.encode(df['Description'])
des_em_en_ = model.encode(df['Description_en'])
```

Normalize the embeddings to mitigate the influence of magnitude on similarity


```python
product_em_tw = normalize_embedding(product_em_tw_)
product_em_en = normalize_embedding(product_em_en_)
des_em_tw = normalize_embedding(des_em_tw_)
des_em_en = normalize_embedding(des_em_en_)
```


```python
product_em = np.concatenate((product_em_tw, product_em_en), axis=1) * alpha[2]
des_em = np.concatenate((des_em_tw, des_em_en), axis=1)  * alpha[3]
text_em = np.concatenate((product_em, des_em), axis=1)
```

Combine category embedding and text embedding


```python
embedding = np.concatenate((category_encode, text_em), axis=1)
embedding.shape
```




    (488, 3108)




```python
df['Sub_Category'].value_counts()
```




    Agritech-Other                                           129
    Agrilivestock-Other                                       50
    Labour-Saving-Machinery-and-Equipment                     40
    Livestock-Feed-and-Additives                              37
    Garden-Materials                                          29
    Sustainable-Aquatic-Products-Other                        24
    Intelligent-Detection-System-and-Equipment                23
    Ventilation-Equipment                                     21
    Organic-Fertilizer                                        15
    Agrifresh-Other                                           13
    Agricultural-Processing-Machinery                         12
    Seedlings-and-Flower-Seed                                 11
    Plant-Disease-and-Pest-Control                            11
    AIoT-Intelligent-Cold-Chain-Logistic-Solution             10
    Refrigeration-and-Freezing-Equipment                      10
    Fish-Farming-Other                                         9
    Intelligent-Temperature-Control-Technology                 8
    AIoT-Equipment-and-System                                  8
    Intelligent-Environmental-Control-Devices-and-Systems      5
    Grow-Light                                                 4
    Agricultural-Automation-Equipment                          4
    Intelligent-Irrigation-System-and-Equipment                3
    Water-Quality-Improve                                      3
    Biotechnology-Applications                                 2
    Feed-and-Feed-Additive                                     1
    Feed-Processing-Equipment-and-Testing-Equipment            1
    Organic-Waste-Disposal-Technology-and-Equipment            1
    Dehydrated-and-Pickled-Aquatic-Products                    1
    Artificial-Fog-Equipment                                   1
    Aquaculture-Technology-and-Management                      1
    Thermal-Camera                                             1
    Name: Sub_Category, dtype: int64



# Recommendation Model Building

## Experiment and Parameter Tuning

Sample items in each category to evaluate


```python
# idx_list = []
# for colname in df['Sub_Category'].unique():
#     condition = df['Sub_Category'] == colname
#     idx = df[condition].sample(n=1).index[0]
#     idx_list.append(idx)
    
# sample_df = df.iloc[idx_list].drop(columns=['Product_id','Vendor_id','Country'])
```

Use all items to evaluate


```python
# idx_list = np.arange(0, df.shape[0])
```

Recommend items by calculating the cosine similarity between the item embeddings. Output the results into a cell file and save them in an Excel file.


```python
# output_file = './Data/Rec-Agr-{}.xlsx'.format(arr)
# blank_row_color = 'FFFF00'  # Yellow color code

# # Load the existing workbook or create a new one
# try:
#     workbook = openpyxl.load_workbook(output_file)
# except FileNotFoundError:
#     workbook = openpyxl.Workbook()

# # Save the workbook
# workbook.save(output_file)

# # Get the default sheet name (usually "Sheet1")
# sheet_name = workbook.sheetnames[0]

# # Create a Pandas Excel writer using openpyxl engine and append mode
# writer = pd.ExcelWriter(output_file, engine='openpyxl', mode='a')

# # Assign the existing workbook to the writer
# writer.book = workbook

# # Select the default sheet
# writer.sheets = {sheet_name: workbook[sheet_name]}

# # Keep track of the row index for writing data
# current_row = writer.sheets[sheet_name].max_row + 1

# # Generate recommendations for each sample index
# for sample_idx in idx_list:
#     row_idx = []
#     score_ls = []
#     rec_k_dic = util.semantic_search(embedding[sample_idx], embedding, top_k=10)[0]
#     for item in rec_k_dic:
#         row_idx.append(item['corpus_id'])
#         score_ls.append(round(item['score'],3))  
# #     select_idx = np.insert(row_idx, 0, sample_idx) # Put the input item as a reference
#     output_df = df.iloc[row_idx]
#     output_df['Score'] = score_ls

#     # Set the cell color for the first row behind the blank row
#     sheet = writer.book[sheet_name]
#     for col in range(1, len(output_df.columns) + 10):
#         cell = sheet.cell(row=current_row + 1, column=col)
#         cell.fill = openpyxl.styles.PatternFill(fill_type='solid', fgColor=blank_row_color)

#     # Write the DataFrame to Excel
#     output_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=True, header=False)

#     # Update the current row index for the next set of recommendations
#     current_row += len(output_df) + 1  # Add 1 for the blank row

# # Save the Excel file
# writer.save()
# workbook.close()
# writer.close()
```

## Recommend Top K


```python
# product_id = 'CU0004601801' # tape
product_id = 'CU0009108101' # alcohol
# product_id = 'CU0004414408' # thermometer


k = 5
threshold = 0.75
row_index = df[df['Product_id'] == product_id].index
df.iloc[row_index]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_id</th>
      <th>Product_Name</th>
      <th>Vendor_id</th>
      <th>Main_Category</th>
      <th>Sub_Category</th>
      <th>Description</th>
      <th>Product_Name_en</th>
      <th>Description_en</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>CU0009108101</td>
      <td>霹靂穠研酒藏</td>
      <td>CU00091081</td>
      <td>sustainable-aquatic-products</td>
      <td>Sustainable-Aquatic-Products-Other</td>
      <td>興藝峰生技農業和霹靂布袋戲，在地深耕，放眼世界，均為台灣原創精神代表！ 跨界聯名推出【霹靂穠...</td>
      <td>Pili &amp; Viachi Spirits Classics</td>
      <td>Agri-Dragon Biotech and Pili International Mul...</td>
    </tr>
  </tbody>
</table>
</div>




```python
rec_pd_id_ls, rec_score_ls, rec_row_idx = recommend_top_k(product_id, k, 0.75)
for i in range(len(rec_pd_id_ls)):
    print('Product_id:', rec_pd_id_ls[i],'Score:', rec_score_ls[i])
df.iloc[rec_row_idx]
```

    Product_id: CU0009108102 Score: 0.811
    Product_id: CU0009108107 Score: 0.807
    Product_id: CU0009108106 Score: 0.796
    Product_id: CU0009108115 Score: 0.781
    Product_id: CU0009108109 Score: 0.78
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_id</th>
      <th>Product_Name</th>
      <th>Vendor_id</th>
      <th>Main_Category</th>
      <th>Sub_Category</th>
      <th>Description</th>
      <th>Product_Name_en</th>
      <th>Description_en</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>CU0009108102</td>
      <td>時氣純韻(糙米白酒)</td>
      <td>CU00091081</td>
      <td>sustainable-aquatic-products</td>
      <td>Sustainable-Aquatic-Products-Other</td>
      <td>天癒糙米淬釀而成，純米風味，口感温順。(32%)</td>
      <td>Viachi Spirit – Longevity (Brown Rice)</td>
      <td>Made from Viachi brown rice, it is pleasantly ...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CU0009108107</td>
      <td>和氣元酒(丹參)</td>
      <td>CU00091081</td>
      <td>sustainable-aquatic-products</td>
      <td>Sustainable-Aquatic-Products-Other</td>
      <td>介紹：天癒醲系列經典代表作。天癒丹參高純度產製，蔘味醇厚，豪氣順喉。鮮採後，輔以獨特製酒技術...</td>
      <td>Viachi Spirit – The Dove (Salvia)</td>
      <td>Introduction：\n \n Grown in the AgriDragon Bio...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CU0009108106</td>
      <td>福氣永豐(葡萄酒)</td>
      <td>CU00091081</td>
      <td>sustainable-aquatic-products</td>
      <td>Sustainable-Aquatic-Products-Other</td>
      <td>天癒巨峰葡萄醇釀，濃烈豐厚，温潤純粹，值得品味珍藏。 \n \n 無添加人工香料及食用酒精\...</td>
      <td>Viachi Spirit – Abundance (Grape)</td>
      <td>Made from healthy grapes cultivated using the ...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CU0009108115</td>
      <td>天癒花草茶</td>
      <td>CU00091083</td>
      <td>sustainable-aquatic-products</td>
      <td>Sustainable-Aquatic-Products-Other</td>
      <td>純淨天然放鬆‧舒緩‧平和\n 內容物：\n 檸檬香蜂草、巧克力薄荷、綠薄荷、檸檬馬鞭草、甜菊...</td>
      <td>Viachi Herbal Tea</td>
      <td>100% Pure &amp; Natural\n \n Relaxing, Calming, an...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CU0009108109</td>
      <td>天癒丹參勁花茶</td>
      <td>CU00091081</td>
      <td>sustainable-aquatic-products</td>
      <td>Sustainable-Aquatic-Products-Other</td>
      <td>介紹：以多項專利天癒仿生科技農法栽培之高品質原樣態素材：丹參莖、葉用枸杞、甜菊葉及馬郁蘭莖葉...</td>
      <td>Viachi Energizing Herbal Tea</td>
      <td>nIntroduction：\n \n Consisting of Salvia Milti...</td>
    </tr>
  </tbody>
</table>
</div>



# Future Development

- How can the performance and accuracy of the recommendation model be evaluated and measured in the context of the exhibition industry, considering the absence of explicit labels?
- Overall, the recommendations are generally accurate. However, the model occasionally generates some irrelevant items for specific products.


# Conclusions

The expected outcome is an item-based recommendation model for the dynamic exhibition industry that significantly enhances attendees' ability to find relevant products efficiently. By addressing the difficulties related to unsupervised learning, lack of quantitative features, and reliance on text information, the model aims to identify key product features, effectively utilize them for recommendations, and employ suitable similarity analysis techniques. Through these methods, the model strives to provide accurate and relevant recommendations, improving the overall experience for attendees.
