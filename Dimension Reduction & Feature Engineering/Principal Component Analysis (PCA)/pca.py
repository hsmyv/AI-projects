# Lab: Əsas Komponent Analizinin (PCA) tətbiqləri

# Giriş
# Bu laboratoriyada PCA-nın iki mühüm tətbiqini öyrənəcəksiniz.

# Birinci tətbiqdə 2 ölçülü verilənləri əsas komponent oxlarına proyeksiya etməyi göstərəcəksiniz, 
# yəni verilənlərdəki dəyişkənliyi ən yaxşı şəkildə izah edən iki ortoqonal istiqamətə.

# İkinci tətbiqdə isə daha çox ölçülü verilənləri daha aşağı ölçülü fəzaya proyeksiya edəcəksiniz.
# Bu ölçü azaldılması nümunəsidir – bu, həm hesablama yükünü azaldır, həm də bir çox halda modelin dəqiqliyini artırır.
# PCA artıq və xətti əlaqəli dəyişənləri çıxarmaqda və məlumatdakı səs-küyü azaltmaqda faydalıdır.


# I Hissə: PCA istifadə edərək 2D verilənləri əsas komponent oxlarına proyeksiya etmək

# Burada PCA-nın 2D verilənləri dəyişkənliyi ən yaxşı izah edən əsas komponent oxlarına çevirmək üçün necə istifadə olunduğunu göstərəcəksiniz.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


# Təsadüfi toxumu təyin edirik ki, nəticələr hər zaman eyni olsun
np.random.seed(42)

# Orta və kovarians matrisini təyin edirik
mean = [0, 0]
cov = [[3, 2], [2, 2]]

# Bivariate normal bölgü əsasında 200 nümunə yaradırıq
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)

# Verilənləri iki ölçülü səpələmə qrafikində göstəririk
plt.figure()
plt.scatter(X[:, 0], X[:, 1],  edgecolor='k', alpha=0.7)
plt.title("İki Ölçülü Normal Paylanmanın Qrafiki")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid(True)
# plt.show()


# Verilənlər üzərində PCA tətbiq edirik
# 2 komponentli PCA obyektini yaradırıq və fit_transform ilə həm öyrədir, həm də transformasiya edirik

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Əsas komponentləri əldə edirik
components = pca.components_

# Hər bir komponentin izah etdiyi dispersiya nisbətini göstəririk
pca.explained_variance_ratio_


# İndi isə nəticələri göstəririk
# Əvvəlcə verilənləri orijinal fəzada səpələmə qrafiki ilə göstəririk

# Sonra isə verilənlərin əsas komponent oxları üzrə proyeksiyalarını çəkəcəyik

# Bu texniki cəhətdən xətti cəbr tələb edir, amma nəticə çox faydalı olacaq


# Verilənləri əsas komponent oxlarına proyeksiya edirik
# Bu proyeksiya nöqtələrin əsas komponent istiqamətində olan koordinatlarını verir

# Koordinatlar verilmiş komponentlə nöqtənin skalyar hasilidir (dot product)

projection_pc1 = np.dot(X, components[0])
projection_pc2 = np.dot(X, components[1])

# Proyeksiya nəticəsində yeni koordinatları hesablamaq
x_pc1 = projection_pc1 * components[0][0]
y_pc1 = projection_pc1 * components[0][1]
x_pc2 = projection_pc2 * components[1][0]
y_pc2 = projection_pc2 * components[1][1]

# Qrafiki çəkirik
plt.figure()
plt.scatter(X[:, 0], X[:, 1], label='Orijinal Verilər', ec='k', s=50, alpha=0.6)

# PC1 və PC2 üzərindəki proyeksiyaları göstəririk
plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='PC1 üzərinə proyeksiya')
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='PC2 üzərinə proyeksiya')
plt.title('Xətti Əlaqəli Verilərin Əsas Komponentlərə Proyeksiyası')
plt.xlabel('Özəllik 1')
plt.ylabel('Özəllik 2')
plt.legend()
plt.grid(True)
plt.axis('equal')
# plt.show()

# Gördüyünüz kimi, əsas komponentlər verilənlərdəki dəyişkənliyi ən yaxşı təmsil edən istiqamətlərdir

# Qırmızı istiqamət (PC1) – dəyişkənliyin ən çox olduğu istiqamətdir













############################################
# Part II. PCA for feature space dimensionality reduction

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA tətbiq et və 2 komponentə endir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA-transformed data in 2D
plt.figure(figsize=(8,6))

colors = ['navy', 'turquoise', 'darkorange']
lw = 1

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=50, ec='k', alpha=0.7, lw=lw,
                label=target_name)

plt.title('IRIS datasının PCA ilə 2 ölçüyə endirilməsi')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio for each component
plt.figure(figsize=(10,6))
plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio, alpha=1, align='center',
        label='PC izah edilən varyans nisbəti')
plt.ylabel('İzah Edilən Varyans Nisbəti')
plt.xlabel('Əsas Komponentlər')
plt.title('Əsas Komponentlər üzrə izah edilən varyans')

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.step(range(1, 3), cumulative_variance, where='mid', linestyle='--', lw=3, color='red',
         label='Yığılı izah edilən varyans')
plt.xticks(range(1, 3))
plt.legend()
plt.grid(True)
plt.show()
