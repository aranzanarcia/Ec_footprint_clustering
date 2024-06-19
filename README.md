# Ec_footprint_clustering
Table of Contents

- Introduction
- Dataset Description
- Data Cleaning and Preparation
- Methodology
- Analysis of Clusters
- Naming of Clusters
- Conclusion
- Presentation

# Introduction
This project aims to analyze ecological footprint data using unsupervised machine learning techniques. The primary goal is to group countries based on their ecological damage and understand the characteristics of each cluster. By leveraging the k-means clustering algorithm, we can identify patterns and similarities among different countries' ecological impact and demographics.

# Dataset Description
The dataset used in this project contains various ecological indicators and some demographic information for different countries. Key features include:

- Country
- Total Ecological Footprint
- Total Biocapacity
- Ecological Deficit or Reserve
- GDP per capita
- Life expectancy
- Income Group

# Data Cleaning and Preparation
The dataset underwent several preprocessing steps to ensure data quality and compatibility with the k-means algorithm:

- Handling Missing Values: Imputed and removed missing values to maintain data integrity.
- Categorical to Numerical Conversion: Mapped the categorical income group to numerical values.
- Normalization: Scaled the data to ensure each feature contributed equally to the clustering process.
- Feature Selection: Selected relevant features for the clustering analysis.

  
# Methodology
1. K-Means Clustering: Applied the k-means algorithm to cluster the countries based on their ecological impact and demographics.
2. Determining Optimal Clusters: Used methods such as the Elbow Method to determine the optimal number of clusters.
3. Cluster Assignment: Assigned each country to a cluster and added a new column "cluster" to the dataset.


# Analysis of Clusters
After clustering the countries, I analyzed each cluster to understand the common characteristics shared by the countries within them. This involved examining the mean values and distributions of key ecological and demographic indicators for each cluster.

# Naming of Clusters
Based on the analysis, I named each cluster to reflect its characteristics and demographic attributes:

1. Greedy:
- Description: This cluster comprises rich countries with high ecological footprints and significant environmental damage.
- Characteristics: These countries have high GDPs and belong to higher income groups. They tend to consume more resources and generate more waste, leading to a larger ecological footprint.

2. Modest:
- Description: This cluster consists of poorer countries with lower ecological footprints.
- Characteristics: These countries have lower GDPs and belong to lower income groups. They consume fewer resources and have a smaller environmental impact compared to other clusters.

3. In-Between:
- Description: Countries in this cluster have somewhat of a high ecological footprints and fall between the "Greedy" and "Modest" clusters.
- Characteristics: These countries have mid-range GDPs and income levels. 

4. In-Between Ideal:
- Description: This cluster is similar to the "In-Between" cluster in terms of demographics, but the countries in this group are the most eco-friendly overall.
- Characteristics: Countries in this cluster also have mid-range GDPs and belong to higher income groups than the "Modest" cluster but are less wealthy than the "Greedy" cluster. Despite this, they manage their resources better and have implemented practices that lead to a significantly lower ecological deficit.


# Conclusion
This project demonstrates the utility of unsupervised machine learning in environmental and demographic data analysis. By clustering countries based on their ecological impact and demographics, we can identify patterns and suggest improvements for better ecological practices, as well as further analysing ecological policies and practices in the "in-between ideal" countries to understand what is making them less ecologically damaging than other clusters.

# Presentation
  
![Captura de pantalla (87)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/3f21bbb4-71f1-420b-ad8e-6daee86b1898)
![Captura de pantalla (88)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/c7433cf7-4353-4d5a-a0cc-b218051ed380)
![Captura de pantalla (89)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/8606a1cf-f853-4f39-ae8d-606b81e941b0)
![Captura de pantalla (90)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/f8871f76-3874-4c1e-88e2-6cd510cd8cca)
![Captura de pantalla (91)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/a7ce97af-5113-47b6-a2ca-fa00bb067907)
![Captura de pantalla (92)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/064481f1-baf7-4697-9dbe-5971f672596e)
![Captura de pantalla (93)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/335081a4-715f-4bf7-a787-db12e9230dad)
![Captura de pantalla (94)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/ebee2a40-c2dc-4474-8560-20519998b85e)
![Captura de pantalla (95)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/099b1acf-757c-4ff8-984b-dc7ad98c6653)
![Captura de pantalla (96)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/06354a62-a7ac-4a81-ba94-d6840342f529)
![Captura de pantalla (97)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/f0027c92-cc82-404f-ad47-0eece884cea7)
![Captura de pantalla (99)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/0a0a5b53-ff89-48a6-aec2-9524f11e318e)
![Captura de pantalla (100)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/6092d71b-255d-4d92-a6fb-cbf753efe4e1)
![Captura de pantalla (102)](https://github.com/aranzanarcia/Ec_footprint_clustering/assets/165634773/ec2a3a5a-3c04-467a-8371-08c0b11cf498)





