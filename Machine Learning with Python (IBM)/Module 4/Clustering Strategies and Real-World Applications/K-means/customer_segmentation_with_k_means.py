import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px


import warnings
warnings.filterwarnings('ignore')




# Müştəri seqmentasiyası - k-means ilə
# Təsəvvür edin ki, sizdə bir müştəri məlumat bazası var və bu məlumatlara əsaslanaraq müştəriləri qruplaşdırmaq (seqmentasiya) istəyirsiniz.
# Seqmentasiya, oxşar xüsusiyyətlərə malik müştəriləri eyni qrupa ayırmaq deməkdir.
# Bu üsul, şirkətlərin marketinq strategiyalarını düzgün istiqamətləndirməsinə və daha effektiv istifadə etməsinə imkan verir.

# Məsələn, yüksək gəlirli və sadiq müştərilər bir qrupa daxil ola bilər, bu qrup üçün ayrıca strategiya hazırlana bilər.

cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")


# Dataset-dəki `Address` (ünvan) sütunu kateqorial dəyişəndir.
# k-means alqoritmi isə yalnız ədədi dəyərlərlə işləyə bilir. Kateqorial dəyərlərlə işləmək üçün onları bir neçə 0 və 1-lə kodlaşdırmaq (one-hot encoding) lazımdır.
# Sadəlik üçün biz `Address` sütununu siləcəyik və onunla işləməyəcəyik.

cust_df = cust_df.drop('Address', axis=1)

# Hər ehtimala qarşı datasetdəki boş (NaN) dəyərləri də silirik.
cust_df = cust_df.dropna()
cust_df.info()



# Normalizasiya – məlumatları eyni ölçüyə salmaq
# Niyə normalizasiya vacibdir? Çünki sahələr fərqli miqyaslarda ola bilər.
# Məsələn, "Gəlir" 1000-lərlə, "Yaş" isə 10-100 arasında dəyişə bilər. 
# Bu zaman "Gəlir" daha çox təsir edər, alqoritm yanıldıcı nəticə verə bilər.
# Normalizasiya ilə bütün sahələri orta dəyəri 0, standart sapması 1 olan miqyaslara gətiririk.

X = cust_df.values[:,1:]  # Customer ID sütununu çıxırıq
Clus_dataSet = StandardScaler().fit_transform(X)


# k-means modeli üçün qrup sayı təyin edirik.
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_


# Alqoritmin yaratdığı etiketləri (hansı müşətrinin hansı qrupta olduğu) `cust_df`-ə əlavə edirik.
cust_df["Clus_km"] = labels


# Qrupların mərkəzlərini analiz etmək üçün hər qrupun orta dəyərlərini çıxarırıq.
# Hər bir xüsusiyyət üzrə orta dəyərlər həmin qrupun "profilini" təyin etməyə kömək edir.
cust_df.groupby('Clus_km').mean()



# Müştərilərin yaş, gəlir və təhsil səviyyəsi üzrə paylanmasına baxaq.
# 2D scatter plot çəkirik: X oxunda yaş, Y oxunda gəlir.
# Nöqtənin ölçüsü təhsili göstərir, rəngi isə qrupları (etiketləri).

area = np.pi * ( X[:, 1])**2  # Təhsil sahəsini böyüdərək sahəni təyin edirik
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
# plt.show()



# İndi isə daha detallı analiz üçün 3D qrafik çəkirik:
# X oxu: Təhsil
# Y oxu: Yaş
# Z oxu: Gəlir
# Nöqtələrin rəngi qrupları göstərir

fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Rəng göstəricisini silir, ölçünü dəyişir

fig.show()

# k-means ilə müştərilər 3 qrupa ayrıldı.
# Hər bir qrupdakı müştərilər bir-birinə demoqrafik olaraq daha çox bənzəyir.

# Tapşırıq:
# Hər bir klasterə profil adı vermək olar. Məsələn:
# 1. GEC KARYERALI, VARLI və TƏHSİLLİ
# 2. ORTA KARYERALI və ORTA GƏLİRLİ
# 3. ERKƏN KARYERALI və AZ GƏLİRLİ
