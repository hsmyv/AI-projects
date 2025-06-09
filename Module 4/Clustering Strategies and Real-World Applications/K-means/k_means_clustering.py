#K-means Clustering

# K-means is vastly used for clustering in many data science applications. It is especially useful if you need to quickly discover insights from unlabeled data.

# Real-world applications of k-means include:

# Customer segmentation
# Understanding what website visitors are trying to accomplish
# Pattern recognition
# Feature engineering
# Data compression


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px


import warnings
warnings.filterwarnings('ignore')



np.random.seed(0)

# -------------------------------------
# TƏSVİRLİ KLASTERLƏRİN YARADILMASI
# -------------------------------------

# make_blobs funksiyası ilə təsadüfi klasterlər yaradacağıq.
# Bu funksiyaya bir neçə parametr verəcəyik:

# n_samples: Klasterlərə bərabər şəkildə bölünəcək nöqtələrin ümumi sayı.
# Dəyər: 5000

# centers: Klaster mərkəzlərinin sayı və ya konkret koordinatları.
# Dəyər: [[4, 4], [-2, -1], [2, -3],[1,1]] — yəni 4 fərqli mərkəz

# cluster_std: Klasterin yayılma (standart sapma) dərəcəsi.
# Dəyər: 0.9 — yəni klasterlər çox dağılmamış olacaq.

# Nəticə:
# X — (nümunə sayı, xüsusiyyət sayı) ölçülü massiv (xüsusiyyət matrisidir)
# y — hər bir nümunənin hansı klasterə aid olduğunu göstərən etiketlər

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# -------------------------------------
# VERİLƏNLƏRİN QRAFİK TƏSVİRİ
# -------------------------------------

# X nöqtələrinin (yəni xüsusiyyətlərin) scatter plot ilə vizuallaşdırılması.
# Bu, yaradılmış verilənlərin necə paylandığını göstərir.

plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, ec='k', s=80)

# -------------------------------------
# K-MEANS KLASTERLƏŞDİRMƏNİN QURULMASI
# -------------------------------------

# İndi təsadüfi yaradılmış bu verilənlər üzərində KMeans klasterləşdirmə tətbiq edəcəyik.

# İstifadə olunan əsas parametrlər:
# init: Mərkəzlərin ilkin seçilmə üsulu.
# "k-means++": KMeans-in daha sürətli və daha yaxşı nəticələr əldə etməsi üçün ağıllı mərkəz seçmə üsuludur.

# n_clusters: Klaster sayı. (Mərkəz sayı da bu qədər olacaq.)
# Dəyər: 4 (çünki biz 4 mərkəz göstərmişdik)

# n_init: Fərqli ilkin mərkəzlərlə neçə dəfə təkrar işləyəcək.
# Yəni ən yaxşı nəticəni əldə etmək üçün 12 dəfə cəhd edəcək.

# KMeans modelini göstərilən parametrlərlə qururuq.
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# Modeli xüsusiyyətlər matrisimizə (X) uyğunlaşdırırıq (öyrədirik).
k_means.fit(X)

# -------------------------------------
# KLASTERLƏRİN NƏTİCƏLƏRİNİ ALMAQ
# -------------------------------------

# Hər bir nöqtənin aid olduğu klasteri (yəni etiketini) əldə edirik.
k_means_labels = k_means.labels_

# Hər bir klasterin koordinatlarını (mərkəz nöqtələrini) əldə edirik.
k_means_cluster_centers = k_means.cluster_centers_

# -------------------------------------
# KLASTERLƏRİN VƏ MƏRKƏZLƏRİN VİZUAL GÖSTƏRİLMƏSİ
# -------------------------------------

# Plot (qrafik) sahəsi yaradılır. Ölçüsü 6x4 olacaq.
fig = plt.figure(figsize=(6, 4))

# Rənglər: Etiket sayına görə avtomatik fərqli rənglər seçilir.
# plt.cm.tab10 — 10 fərqli rəngi təklif edən colormap-dır.
# np.linspace — rəngləri bərabər şəkildə bölmək üçün istifadə olunur.
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# Əsas subplot yaradılır.
ax = fig.add_subplot(1, 1, 1)

# İndi 4 klasterin hamısını dövrə alıb çəkəcəyik.
# `k` hər dövrdə bir klasterin indeksini göstərir (0-dan 3-ə qədər).
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Bu klasterə aid olan nöqtələri seçirik.
    my_members = (k_means_labels == k)

    # Bu klasterin mərkəz nöqtəsini götürürük.
    cluster_center = k_means_cluster_centers[k]

    # Bu klasterə aid nöqtələri rənglə çəkirik.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', ms=10)

    # Klasterin mərkəz nöqtəsini çəkirik (qara kənarlıqla).
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Qrafikin başlığı
ax.set_title('KMeans')

# X oxundakı rəqəmləri göstərmirik
ax.set_xticks(())

# Y oxundakı rəqəmləri göstərmirik
ax.set_yticks(())

# Qrafiki göstəririk
# plt.show()



















# Exercise 1
# Try to cluster the above dataset into a different number of clusters, say k=3. Note the difference in the pattern generated.


# k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
# k_means3.fit(X)
# fig = plt.figure(figsize=(6, 4))
# colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means3.labels_))))
# ax = fig.add_subplot(1, 1, 1)
# for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
#     my_members = (k_means3.labels_ == k)
#     cluster_center = k_means3.cluster_centers_[k]
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
# plt.show()













# Exercise 2
# Try the same with k=5.

# k_means3 = KMeans(init="k-means++", n_clusters=5, n_init=12)
# k_means3.fit(X)
# fig = plt.figure(figsize=(6, 4))
# colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means3.labels_))))
# ax = fig.add_subplot(1, 1, 1)
# for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
#     my_members = (k_means3.labels_ == k)
#     cluster_center = k_means3.cluster_centers_[k]
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
# plt.show()








#Exercise 3
# Comment on the within-cluster sum of squares, i.e. inertia, of the clusters created for k=3 and k=5.

# For k=3, the value of within-cluster sum of squares will be higher that that for k=4, 
# since the points from different natural clusters are being grouped together, leading to underfitting of the k-means model. 
# For k=5, the value of will be lesser than that for k=4, since the points are distributed into mode clusters than needed, leading to over-fitting of the k-means model.