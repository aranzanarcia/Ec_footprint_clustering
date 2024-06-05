#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from sklearn.cluster import KMeans


# In[2]:


df = pd.read_csv('C:\\Users\\X360\\NOD PROJECTS\\gef23.csv', encoding='iso-8859-1')


# In[3]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df


# In[4]:


df = df.dropna(thresh=len(df.columns) - 2)


# In[5]:


df.info()


# In[6]:


percgdp= {"Afghanistan": 356, "Cuba": 9499.6,"Lebanon": 4136.1, "Switzerland": 93259.9, "Syrian Arab Republic": 420.6, "Uganda":964.4, "Uzbekistan":2255.2, "Venezuela, Bolivarian Republic of" :15975.7}
#sdgi = {"Equatorial Guinea": , "Eritrea": , "Guinea-Bissau": , "Saint Lucia": , "Slovenia": , "Thailand": ,"Togo": , "Uruguay":}


# In[7]:


df.loc[df['Country'] == 'Cuba', 'Per Capita GDP'] = 9499.6
df.loc[df["Country"] == "Côte d'Ivoire", "Life Exectancy"] = 59
df.loc[df["Country"] == "Côte d'Ivoire", "HDI"] = 0.55
df.loc[df["Country"] == "Somalia", "HDI"] = 0.285
df.loc[df["Country"] == "Côte d'Ivoire", "Per Capita GDP"] = 2486.41
df.loc[df["Country"] == "Côte d'Ivoire", "SDGi"] = 120


# In[8]:


df['Per Capita GDP'] = df['Per Capita GDP'].str.replace('$', '').str.replace(',', '').astype(float)
df['Population (millions)'] = df['Population (millions)'].str.replace(',', '').astype(float)


# In[9]:


def fill_per_capita_gdp(row):
    if pd.isna(row['Per Capita GDP']):
        return percgdp.get(row['Country'], row['Per Capita GDP'])
    return row['Per Capita GDP']

df['Per Capita GDP'] = df.apply(lambda row: fill_per_capita_gdp(row), axis=1)


# In[10]:


df.loc[df["Country"] == "Côte d'Ivoire", "Per Capita GDP"] = 2486.41


# In[11]:


df.rename(columns = {"Life Exectancy" : "Life Expectancy"}, inplace = True)


# In[12]:


df['Life Expectancy'] = df['Life Expectancy'].astype(float)
df["HDI"] = df["HDI"].astype(float)


# In[13]:


df.info()


# In[14]:


#df.drop(columns = "SDGi", inplace = True)
df = df.drop(df[df['Country'] == 'Suriname'].index)


# In[15]:


income_map = {"LI":1, "LM":2, "UM":3, "HI":4}
df["Income Group"] = df["Income Group"].replace(income_map)


# In[16]:


df.reset_index(inplace=True, drop= True)


# In[17]:


df_nums = df.select_dtypes("number").copy()


# scaler = StandardScaler()
# decomp = PCA()
# pipe = make_pipeline(scaler,decomp)
# 
# pipe.fit(df_nums)
# explained_variance = pipe['pca'].explained_variance_ratio_.cumsum()
# index = [i+1 for i in range(len(explained_variance))]
# 
# fig, ax = plt.subplots(figsize=(12, 8))
# 
# sns.lineplot(x=index, y=explained_variance)
# sns.scatterplot(x=index, y=explained_variance, s=100)
# plt.xlim((1-0.2, len(explained_variance)+0.2))
# plt.ylim((0, 1.1))
# x_s, x_e = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(x_s+0.2, x_e))
# ax.hlines(y=0.9, xmin=1, xmax=len(explained_variance), color='gray', linestyle='--')
# plt.ylabel('Total Explained Variance Ratio')
# plt.xlabel('PC')
# plt.show()

# In[18]:


scaler = StandardScaler()
kmeans = KMeans(n_init=13)

visualizer = KElbowVisualizer(kmeans, k=(2,10))

pipe = make_pipeline(scaler, visualizer)

pipe.fit(df_nums)

# Show the visualizer
visualizer.show()


# In[19]:


scaler = StandardScaler()
#decomp = PCA(n_components = 20)
cluster = KMeans(n_init=25, n_clusters =4)
pipe = make_pipeline(scaler, cluster)

pipe.fit(df_nums)


# pipe.fit(df_nums)
# values = pipe[:2].transform(df_nums)
# pca_labels = [f"PC{idx+1}" for idx, i in enumerate(values.T)]
# 
# df = df.join(pd.DataFrame(values, columns=pca_labels))

# In[20]:


df["clusters"] = pipe['kmeans'].labels_
df


# In[21]:


df["clusters"].value_counts()


# In[22]:


df.groupby("clusters")[["HDI", "Per Capita GDP", "Income Group", "Carbon Footprint", "Total Ecological Footprint (Consumption)","Forest Product Footprint", "Total biocapacity ", "Ecological (Deficit) or Reserve", "Number of Earths required"]].agg(["mean", "max", "min"])


# In[23]:


df


# In[24]:


df.groupby("clusters")["Region"].value_counts()


# In[25]:


df.groupby("clusters")[["Per Capita GDP", "Total Ecological Footprint (Consumption)", "Life Expectancy", "Total biocapacity ", "Ecological (Deficit) or Reserve"]].agg(["mean", "min", "max"])


# In[ ]:


df.groupby("clusters")["Population (millions)"].agg(["mean", "min", "max"])


# In[ ]:


df.groupby("clusters")["Income Group"].value_counts()


# In[ ]:


clusters_mapping = {0: "Desirable", 1:"Most Damaging", 3:"Least damaging", 2:"Suriname (Most biocapacity)"}
df["clusters"] = df["clusters"].replace(clusters_mapping)


# In[ ]:


df.groupby("clusters")[["Per Capita GDP", "Total Ecological Footprint (Consumption)", "Life Expectancy", "Total biocapacity ", "Ecological (Deficit) or Reserve"]].mean()


# In[ ]:


# 2 is the in-between ideal
# 3 is in between
# 1 greedy 
# 0 poor 


# In[26]:


import geopandas as gpd
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

geo_countries = world['name'].tolist()

# List to store non-matching country names
non_matching_countries = []

# Loop through the countries in the dataframe and check for matches
for country in df['Country']:
    if country not in geo_countries:
        non_matching_countries.append(country)
non_matching_countries


# In[27]:


countries_mapping = {"Turkiye":"Turkey","Syrian Arab Republic":"Syria","Tanzania, United Republic of":"Tanzania","South Sudan":"S. Sudan","Russian Federation":"Russia", "Equatorial Guinea":"Eq. Guinea","Lao People's Democratic Republic":"Laos", "Iran, Islamic Republic of":"Iran","Korea, Republic of":"South Korea", "Dominican Republic":"Dominican Rep.","Czech Republic":"Czechia", "Brunei Darussalam":"Brunei","Central African Republic":"Central African Rep.","Republic of North Macedonia":"North Macedonia", "Congo, Democratic Republic of":"Dem. Rep. Congo", "Dominican Republic": "Dominican Rep.", "Venezuela, Bolivarian Republic of": "Venezuela",
    "Bosnia and Herzegovina": "Bosnia and Herz.", "Viet Nam" : "Vietnam", "Eswatini": "eSwatini"}


# In[28]:


df['Country'] = df['Country'].replace(countries_mapping)

# Ensure country names match for merging
df['Country'] = df['Country'].str.strip()


# In[243]:


import geopandas as gpd
import matplotlib.pyplot as plt

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge the world dataframe with the data dataframe
world = world.merge(df, how="left", left_on="name", right_on="Country")

# Define the cluster colors
cluster_colors = {
    0: "#c47aa3",
    1: "#b94651",
    2: "#5dc03f",
    3: "#4ea8cb"
}

# Map the cluster colors, filling NaN values with a default color
world['cluster_color'] = world['clusters'].map(cluster_colors)
world['cluster_color'] = world['cluster_color'].fillna("#e8e2e6")  # Use a light gray color for NaN values

# Descriptive labels for the clusters
label_mapping = {
    0: "Modest",
    1: "Greedy",
    2: "In-between Ideal",
    3: "In-between"
}

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot boundaries without grid lines
world.boundary.plot(ax=ax)

# Plot countries with cluster colors
world.plot(color=world['cluster_color'], ax=ax, legend=False)

# Create a custom legend with bigger size
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[label], markersize=10) for label in cluster_colors.keys()]
legend_labels = [label_mapping[label] for label in cluster_colors.keys()]
legend = ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1, 1), loc='upper left', title='Clusters', fontsize='large')

# Set legend background color to white
legend.get_frame().set_facecolor('white')

# Set legend border color to black
legend.get_frame().set_edgecolor('black')

# Add a title
plt.title("CLUSTER COUNTRIES", fontsize = 16)

# Show the plot
plt.savefig('cluster_map_.png', transparent = True)
plt.show()


# In[31]:


region_counts = df.groupby(['clusters', 'Region']).size().unstack(fill_value=0)

# Calculate the percentages within each cluster
region_percentages = region_counts.div(region_counts.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plotting each cluster
for i, cluster in enumerate(region_percentages.index):
    ax = axes[i]
    region_percentages.loc[cluster].plot(kind='bar', ax=ax, color=sns.color_palette("tab20"))
    ax.set_title(f'Regional Distribution in Cluster: {label_mapping[cluster]}')
    ax.set_xlabel('Region')
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 100)

# Remove any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# In[74]:


for cluster in region_percentages.index:
    cluster_label = label_mapping[cluster]
    
    # Filter out zero values
    non_zero_percentages = region_percentages.loc[cluster][region_percentages.loc[cluster] > 0]
    
    plt.figure(figsize=(8, 8))
    plt.pie(non_zero_percentages, labels=non_zero_percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Regional Distribution in Cluster: {cluster_label}')
    plt.savefig(f"{cluster}_clusterreg.png", transparent = True)
    plt.show()


# In[33]:


df


# In[79]:


mean_values = df.groupby('clusters')['Ecological (Deficit) or Reserve'].mean().sort_values()

# Define colors for the bars
colors = ["#b3b6bd"] * 3 + ["#6EB6E0"] * 1

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="Ecological (Deficit) or Reserve",
    errorbar=None,
    estimator="mean",
    order=mean_values.index,
    palette=colors
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_values.index], fontsize=12)

# Add numerical values above or inside the bars
for index, value in enumerate(mean_values):
    ax1.text(index, value / 2, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')


# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("MEAN ECOLOGICAL RESERVE (+)\n & DEFICIT(-) PER CLUSTER", fontsize=14)
ax1.set_xlabel(None)
ax1.set_ylabel(None)

# Show plot
plt.savefig("ec_reserve.png", transparent = True)
plt.show()


# In[80]:


df.groupby('clusters')['Life Expectancy'].mean()


# In[84]:


mean_val_le = df.groupby('clusters')['Life Expectancy'].mean().sort_values()

# Define colors for the bars
colors = ["#b3b6bd"] * 1 + ["#6EB6E0"] * 2 + ["#b3b6bd"] * 1

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="Life Expectancy",
    errorbar=None,
    estimator="mean",
    order=mean_val_le.index,
    palette=colors
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_val_le.index], fontsize=12)

# Add numerical values above or inside the bars
for index, value in enumerate(mean_val_le):
    ax1.text(index, value / 2, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')


# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("MEAN LIFE EXPECTANCY PER CLUSTER", fontsize=14)
ax1.set_xlabel(None)
ax1.set_ylabel(None)

# Show plot
plt.savefig("life_exp.png", transparent = True)
plt.show()


# In[99]:


mean_val_le = df.groupby('clusters')['Life Expectancy'].mean().sort_values()
min_val_le = df.groupby('clusters')['Life Expectancy'].min().sort_values()
max_val_le = df.groupby('clusters')['Life Expectancy'].max().sort_values()

# Define colors for the bars
colors = ["#b3b6bd"] * 1 + ["#6EB6E0"] * 2 + ["#b3b6bd"] * 1

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="Life Expectancy",
    estimator="mean",
    order=mean_val_le.index,
    palette=colors
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_val_le.index], fontsize=12)

# Add numerical values above the bars
for index, value in enumerate(mean_val_le):
    ax1.text(index, value / 2, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')

# Add numerical min and max values below the bars
for index, (mean, min_val, max_val) in enumerate(zip(mean_val_le, min_val_le, max_val_le)):
    ax1.text(index, min_val - 1, f'Min: {min_val:.2f}', ha='center', va='top', fontsize=8, color='black')
    ax1.text(index, max_val + 1, f'Max: {max_val:.2f}', ha='center', va='bottom', fontsize=8, color='black')

# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("MEAN LIFE EXPECTANCY PER CLUSTER", fontsize=14, y=1.05)  # Adjust y position of the title
ax1.set_xlabel(None)
ax1.set_ylabel(None)

# Show plot
plt.savefig("life_expect.png", transparent=True)
plt.show()


# In[100]:


df['SDGi'] = pd.to_numeric(df['SDGi'], errors='coerce')


# In[103]:


df.groupby('clusters')['SDGi'].mean()


# In[106]:


df


# In[247]:


sorted_label_mapping = {key: label_mapping[key] for key in sorted(label_mapping.keys())}
custom_palette = {
    0: "#c47aa3",
    1: "#b94651",
    2: "#5dc03f",
    3: "#4ea8cb"
}
# Set the style of the seaborn plot
sns.set(style="whitegrid")
sns.despine(top=True, bottom = True, left=True)

# Create the scatter plot with custom palette
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total Ecological Footprint (Consumption)', y='Total biocapacity ', data=df, hue='clusters', palette=custom_palette)

# Set labels and title
plt.xlabel('ECOLOGICAL FOOTPRINT', fontsize=12)
plt.ylabel('BIOCAPACITY', fontsize=12)
plt.title('BIOCAPACITY AND ECOLOGICAL\nFOOTPRINT BY CLUSTER', fontsize=14)

# Create legend handles and labels
legend_handles = [mpatches.Patch(color=custom_palette[key], label=sorted_label_mapping[key]) for key in sorted_label_mapping.keys()]

# Add legend with custom handles and labels
plt.legend(handles=legend_handles, title='Clusters')

# Show plot
plt.savefig("biocap_ecfpt.png", transparent=True)
plt.show()


# In[186]:


from matplotlib.ticker import FuncFormatter

# Map income group codes to descriptive labels
income_group_mapping = {
    1: 'Low Income',
    2: 'Lower-Middle',
    3: 'Upper-Middle',
    4: 'High Income'
}

# Define a custom palette of shades of blue
custom_palette1 = sns.color_palette("Blues", 4)

# Calculate percentage of each income group within each cluster
income_group_percentage = df.groupby(['clusters', 'Income Group']).size() / df.groupby('clusters').size()

# Reset index to make the DataFrame suitable for plotting
income_group_percentage = income_group_percentage.reset_index(name='Percentage')

# Map cluster labels using label_mapping
income_group_percentage['clusters'] = income_group_percentage['clusters'].map(label_mapping)

# Map income group codes to descriptive labels
income_group_percentage['Income Group'] = income_group_percentage['Income Group'].map(income_group_mapping)

# Set the style of the seaborn plot
sns.despine(top=True, bottom=True, left=True)


# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='clusters', y='Percentage', hue='Income Group', data=income_group_percentage, palette=custom_palette1)

# Format the y-axis labels to display percentages
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Set labels and title
plt.xlabel(None)
plt.ylabel('PERCENTAGE', fontsize=12)
plt.title('INCOME GROUPS BY CLUSTER', fontsize=14)

# Show plot
plt.savefig("income_cluster.png", transparent = True)
plt.show()


# In[148]:


df


# In[156]:


df.groupby("clusters")[["Per Capita GDP"]].mean()


# In[155]:


mean_val_hdi = df.groupby('clusters')['HDI'].mean().sort_values()

# Define colors for the bars
colors = ["#b3b6bd"] * 1 + ["#6EB6E0"] * 2 + ["#b3b6bd"] * 1

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="HDI",
    estimator="mean",
    order=mean_val_hdi.index,
    palette=colors, errorbar=None
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_val_hdi.index], fontsize=12)

for index, value in enumerate(mean_val_hdi):
    ax1.text(index, value / 2, f'{value:.3f}', ha='center', va='center', fontsize=10, color='black')


# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("MEAN HUMAN DEVELOPMENT \n INDEX PER CLUSTER", fontsize=14, y=1.05)  
ax1.set_xlabel(None)
ax1.set_ylabel(None)

# Show plot
plt.savefig("hdi_cluster.png", transparent=True)
plt.show()


# In[185]:


mean_val_gdp = df.groupby('clusters')['Per Capita GDP'].mean().sort_values()
# Define colors for the bars
colors = ["#b3b6bd"] * 1 + ["#6EB6E0"] * 2 + ["#b3b6bd"] * 1

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="Per Capita GDP",
    estimator="mean",
    order=mean_val_gdp.index,
    palette=colors,
    errorbar=None
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_val_gdp.index], fontsize=12)

# Add numerical values above the bars
for index, value in enumerate(mean_val_gdp):
    ax1.text(index, value / 2, '{:,.0f}'.format(value), ha='center', va='center', fontsize=10, color='black')


# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("MEAN PER CAPITA GDP BY CLUSTER", fontsize=14, y=1.05)  # Adjust y position of the title
ax1.set_xlabel(None)
ax1.set_ylabel(None)

# Show plot
plt.savefig("gdp_cluster.png", transparent=True)

plt.show()


# In[187]:


df.columns


# In[203]:


df.groupby("clusters")[["Population (millions)", "Forest Product Footprint", "Carbon Footprint","Fish Footprint", "Forest land", "Grazing land", "Number of Countries required", "Number of Earths required"]].mean()


# In[197]:


df.loc[df["Total biocapacity "].idxmax()]


# In[201]:


df["Country"].loc[df["clusters"] == 2]


# In[248]:


sorted_label_mapping = {key: label_mapping[key] for key in sorted(label_mapping.keys())}

# Set the style of the seaborn plot
sns.despine(top=True, bottom = True, left=True)

# Create the scatter plot with custom palette
plt.figure(figsize=(10, 6))
sns.scatterplot(y='Population (millions)', x='Carbon Footprint', data=df, hue='clusters', palette=custom_palette)
plt.ylim(0, 140)
plt.xlim(0, 5)


# Set labels and title
plt.xlabel('CARBON FOORPRINT (PER PERSON)', fontsize=12)
plt.ylabel('POPULATION', fontsize=12)
plt.title('POPULATION AND CARBON FOOTPRINT \n BY COUNTRIES', fontsize=14)

# Create legend handles and labels
legend_handles = [mpatches.Patch(color=custom_palette[key], label=sorted_label_mapping[key]) for key in sorted_label_mapping.keys()]

# Add legend with custom handles and labels
plt.legend(handles=legend_handles, title='Clusters')

# Show plot
plt.savefig("cf_pop1.png", transparent=True)
plt.show()


# In[229]:


mean_val_pop = df.groupby('clusters')['Population (millions)'].mean().sort_values()


# Define colors for the bars
colors = ["#b3b6bd"] * 1 + ["#6EB6E0"] * 1 + ["#b3b6bd"] * 2

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="Population (millions)",
    estimator="mean",
    order=mean_val_pop.index,
    palette=colors,
    errorbar=None
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_val_pop.index], fontsize=12)

# Add numerical values above the bars
for index, value in enumerate(mean_val_pop):
    ax1.text(index, value / 2, '{:,.0f}'.format(value), ha='center', va='center', fontsize=10, color='black')


# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("MEAN POPULATION BY CLUSTER", fontsize=14, y=1.05)  # Adjust y position of the title
ax1.set_ylabel("POPULATION IN MILLIONS")
ax1.set_xlabel(None)

# Show plot
plt.savefig("pop_cluster.png", transparent=True)

plt.show()


# In[225]:


mean_val_cf = df.groupby('clusters')['Carbon Footprint'].mean().sort_values()
min_val_cf = df.groupby('clusters')['Carbon Footprint'].min().sort_values()
max_val_cf = df.groupby('clusters')['Carbon Footprint'].max().sort_values()

# Define colors for the bars
colors = ["#b3b6bd"] * 1 + ["#6EB6E0"] * 1 + ["#b3b6bd"] * 2

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="Carbon Footprint",
    estimator="mean",
    order=mean_val_cf.index,
    palette=colors,
    errorbar=None
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_val_cf.index], fontsize=12)

# Add numerical values above the bars
for index, value in enumerate(mean_val_cf):
    ax1.text(index, value / 2, '{:,.3f}'.format(value), ha='center', va='center', fontsize=10, color='black')
for index, (mean, min_val, max_val) in enumerate(zip(mean_val_cf, min_val_pop, max_val_cf)):
    ax1.text(index, mean + .15, f'Min: {min_val:.2f}', ha='center', va='top', fontsize=8, color='black')
    ax1.text(index, mean + .25, f'Max: {max_val:.2f}', ha='center', va='bottom', fontsize=8, color='black')

# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("MEAN CARBON FOOTPRINT BY CLUSTER", fontsize=14, y=1.05)  # Adjust y position of the title
ax1.set_xlabel(None)
ax1.set_ylabel("CARBON FOOTPRINT (PER PERSON)")

# Show plot
plt.savefig("cf_cluster.png", transparent=True)

plt.show()


# In[227]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='clusters', x='Number of Countries required', hue='clusters', palette='Set2')

# Set labels and title
plt.xlabel('Clusters', fontsize=12)
plt.ylabel('Number of Countries required', fontsize=12)
plt.title('Number of Countries required by Cluster', fontsize=14)

# Show plot
plt.show()


# In[249]:


sorted_label_mapping = {key: label_mapping[key] for key in sorted(label_mapping.keys())}

# Set the style of the seaborn plot
sns.despine(top=True, bottom = True, left=True)

# Create the scatter plot with custom palette
plt.figure(figsize=(10, 6))
sns.scatterplot(x='HDI', y='Ecological (Deficit) or Reserve', data=df, hue='clusters', palette=custom_palette)



# Set labels and title
plt.xlabel('HUMAN DEVELOPMENT INDEX', fontsize=12)
plt.ylabel('ECOLOGICAL RESERVE/DEFICIT', fontsize=12)
plt.title('ECOLOGICAL RESERVE/DEFICIT IN COMPARISON TO HDI', fontsize=14)

# Create legend handles and labels
legend_handles = [mpatches.Patch(color=custom_palette[key], label=sorted_label_mapping[key]) for key in sorted_label_mapping.keys()]

# Add legend with custom handles and labels
plt.legend(handles=legend_handles, title='Clusters')

# Show plot
plt.savefig("ecres_hdi.png", transparent=True)
plt.show()


# In[251]:


sorted_label_mapping = {key: label_mapping[key] for key in sorted(label_mapping.keys())}
custom_palette2 = {
    0: "#bbc6c9",
    1: "#bbc6c9",
    2: "#34a5cb",
    3: "#CB5A34"
}
# Set the style of the seaborn plot
sns.despine(top=True, bottom = True, left=True)

# Create the scatter plot with custom palette
plt.figure(figsize=(10, 6))
sns.scatterplot(y='Population (millions)', x='Carbon Footprint', data=df, hue='clusters', palette=custom_palette2)
plt.ylim(0, 140)
plt.xlim(0, 5)


# Set labels and title
plt.xlabel('CARBON FOORPRINT (PER PERSON)', fontsize=12)
plt.ylabel('POPULATION', fontsize=12)
plt.title('POPULATION AND CARBON FOOTPRINT \n BY COUNTRIES', fontsize=14)

# Create legend handles and labels
legend_handles = [mpatches.Patch(color=custom_palette2[key], label=sorted_label_mapping[key]) for key in sorted_label_mapping.keys()]

# Add legend with custom handles and labels
plt.legend(handles=legend_handles, title='Clusters')

# Show plot
plt.savefig("cf_pop2.png", transparent=True)
plt.show()


# In[268]:


mean_val_nc = df.groupby('clusters')['Number of Countries required'].mean().sort_values()
# Define colors for the bars
colors =  ["#6EB6E0"] * 1 + ["#b3b6bd"] * 3

# Create the barplot
ax1 = sns.barplot(
    data=df,
    x="clusters",
    y="Number of Countries required",
    estimator="mean",
    order=mean_val_nc.index,
    palette=colors,
    errorbar=None
)

# Change x-axis labels to descriptive cluster labels
ax1.set_xticklabels([label_mapping[cluster] for cluster in mean_val_nc.index], fontsize=12)

# Add numerical values above the bars
for index, value in enumerate(mean_val_nc):
    ax1.text(index, value / 2, '{:,.2f}'.format(value), ha='center', va='center', fontsize=10, color='black')


# Remove spines and set titles/labels
sns.despine(top=True, bottom=True, left=True)
plt.title("NUMBER OF COUNTRIES REQUIRED BY CLUSTER", fontsize=14, y=1.05)  # Adjust y position of the title
ax1.set_xlabel(None)
ax1.set_ylabel("MEAN NUMBER OF COUNTRIES")

# Show plot
plt.savefig("nc_cluster.png", transparent=True)

plt.show()


# In[262]:


label_mapping = {0: "Modest", 1: "Greedy", 2: "In-between Ideal", 3: "In-between"}
custom_palette = {"Modest": "#c47aa3", "Greedy": "#b94651", "In-between Ideal": "#4fb08b", "In-between": "#6aa3c3"}

# Create a new column for labeled clusters
df['Cluster Labels'] = df['clusters'].map(label_mapping)

# Set the style of the seaborn plot
sns.set(style="whitegrid")

# Create the box plot with clusters on the y-axis and Number of Earths required on the x-axis
plt.figure(figsize=(12, 8))
sns.boxplot(y='Cluster Labels', x='Number of Earths required', data=df, palette=custom_palette)

# Set labels and title
plt.ylabel('Clusters', fontsize=12)
plt.xlabel('Number of Earths Required', fontsize=12)
plt.title('Distribution of Number of Earths Required by Cluster', fontsize=14)

# Show plot
#plt.savefig("earths_required_by_cluster.png", transparent=True)
plt.show()


# In[ ]:




