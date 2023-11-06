# %%
import pandas as pd

df = pd.read_csv('/Users/astore.ru/Desktop/CLRTAP_original.csv', delimiter="\t")
# %%
df
# %%
df.shape
print(df['Pollutant_name'].unique())
print(df["Country"].unique())
# %%
df = df[df['Pollutant_name'] == 'Pb']
df
# %%
df.to_csv('/Users/astore.ru/Downloads/CLRTAP_Pb_result.csv', index=False)

# %%
# droping emissions summed up for EU and EEA
df = df[(df['Country'] != 'EEA32') & (df['Country'] != 'EU27')]
total_emissions = df['Emissions'].sum()
# %%
total_emissions
# %%
# Calculating summary statistics of "Emissions"
df['Emissions'].min()

# %%
# minimum value that is not zero
import numpy as np

min_nonzero_value = df[df['Emissions'] != 0]['Emissions'].min()
min_nonzero_value
# %%
df['Emissions'].max()

# %%
df['Emissions'].mean()

# %% 2. Explore the data and calculate some summary statistics of the “Emissions”
df_stats = df.groupby("Country")["Emissions"].agg([np.sum, np.min, np.max, np.mean])
df_stats

# %% 3. Check the number of missing values of each variable
# shows which columns contain missing values
import numpy as np

missing_values = df.isnull().any()
missing_values
# %%
# number of missing valueas in each column
number_missiong_values = df.isnull().sum()
number_missiong_values

# %% 4. Create a new geographical variable dividing Europe to four different regions: North, East, South, West.
df.Country.unique()

# %%
data = [['Austria', 'West'], ['Belgium', 'West'], ['Denmark', 'North'], ['Finland', 'North'], ['France', 'West'],
        ['Germany', 'West'],
        ['Greece', 'South'], ['Ireland', 'West'], ['Italy', 'South'], ['Luxembourg', 'West'], ['Netherlands', 'West'],
        ['Portugal', 'South'], ['Spain', 'South'], ['Sweden', 'North'], ['Bulgaria', 'East'], ['Croatia', 'East'],
        ['Cyprus', 'East'],
        ['Czechia', 'East'], ['Estonia', 'East'], ['Hungary', 'East'], ['Lithuania', 'East'], ['Latvia', 'East'],
        ['Malta', 'East'],
        ['Poland', 'East'], ['Romania', 'East'], ['Slovenia', 'East'], ['Slovakia', 'East'], ['EU27', 'n/a'],
        ['Switzerland', 'West'],
        ['Iceland', 'North'], ['Liechtenstein', 'West'], ['Norway', 'North'], ['Türkiye', 'East'],
        ['Czech Republic', 'East'], ['EEA32', 'n/a']]

# %%
country_to_region = {row[0]: row[1] for row in data}

df['Region'] = df['Country'].map(country_to_region)

df

# %% 5. Calculate and print the total sum of emissions (cumulative over years)
yearly_sums = df.groupby('Year')['Emissions'].sum().reset_index()
yearly_sums
cumulative_sums = yearly_sums['Emissions'].cumsum()
cumulative_sums

yearly_sums['Cumulative Sums'] = cumulative_sums
yearly_sums

# %% 6. Calculate the sum of emissions by region and year, make it a new DataFrame
# correct? have less rows

regions_emissions = df[['Region', 'Year', 'Emissions']]
regions_emissions
emissions_by_region = regions_emissions.groupby(['Year', 'Region']).sum().reset_index()
emissions_by_region

# %% 7. Draw line plots of the total emissions of your pollutant in the four European regions and in Europe in total.
import matplotlib.pyplot as plt

regions = ["North", "South", "East", "West"]

for region in regions:
    reg_em = emissions_by_region[emissions_by_region['Region'] == region]
    plt.plot(reg_em['Year'], reg_em['Emissions'], marker='o')
    plt.xlabel('Year')
    plt.ylabel('Total Emissions')
    plt.title(f'Total Emissions by Year for Region {region}')
    plt.show()

# %% plot for Europe in total
plt.plot(yearly_sums['Year'], yearly_sums['Emissions'])
plt.xlabel('Year')
plt.ylabel('Emissions')
plt.title('Total emissions in Europe by year')

# %% Phase III: Create a new DataFrame including only the main sector and excluding all other sectors. Keep all variables (columns).
df_Pb_imp = pd.read_excel('/Users/astore.ru/Downloads/CLRTAP_Pb_result_PHASE3.csv.xlsx')
df_Pb_imp

# %% maybe add third column? Country name or Sector

df_Pb = df_Pb_imp.loc[df_Pb_imp['Country Name'] == "EU27"]
df_Pb = df_Pb.iloc[:, 5:]
df_Pb = df_Pb.transpose().reset_index()
df_Pb.columns = ["Year", "Value"]
df_Pb

# %% find another dataset on production volume or economy
df_GDP_imp = pd.read_csv('/Users/tobias/Documents/test/Project/DATA/GFCF.csv')
df_GDP_imp

# %%
df_GDP = df_GDP_imp.loc[df_GDP_imp['geo'] == 'EU27_2020']
df_GDP = df_GDP[['TIME_PERIOD', 'OBS_VALUE']]
df_GDP.columns = ["Year", "Value"]
df_GDP

# %%
plt.plot(df_Pb['Year'], df_Pb['Value'])
plt.xlabel('Year')
plt.ylabel('Emissions')
plt.title('Pb Residentials')
plt.xticks(rotation=45, ha='right')

# %%
plt.plot(df_GDP['Year'], df_GDP['Value'])
plt.xlabel('Year')
plt.ylabel('Emissions')
plt.title('GDP Residentials')

# %%
# Choose the common time range
common_time_range = range(1995, 2022)

# Filter the data in both DataFrames to match the common time range
df_Pbc = df_Pb[df_Pb['Year'].isin(common_time_range)]
df_GDPc = df_GDP[df_GDP['Year'].isin(common_time_range)]

# %%
# Create a figure and axis for the plot
fig, ax1 = plt.subplots()

# Plot the data from df_1
ax1.plot(df_Pb['Year'], df_Pb['Value'], label='Pb', color='b')
ax1.set_xlabel('Year')
ax1.set_ylabel('Values for Pb', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(0, 1000)
ax1.set_xlim("1995", "2021")

# Create a second y-axis on the same plot
ax2 = ax1.twinx()

# Plot the data from df_2
ax2.plot(df_GDP['Year'], df_GDP['Value'], label='GDP', color='r')
ax2.set_ylabel('Values for GDP', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Set the title and legend
plt.title('Comparison of Pb and GDP')
plt.legend(loc='upper left')

# Show the plot
plt.show()

# %%
fig, ax = plt.subplots()

ax.plot(df_GDPc["Year"], df_GDPc["Value"])
ax.plot(df_Pbc["Year"], df_Pbc["Value"])

# Customize the x-axis label
ax.set_xlabel("Time (months)")

# Customize the y-axis label
ax.set_ylabel("Precipitation (inches)")

# Add the title
ax.set_title("Weather patterns in Austin and Seattle")

# Display the figure
plt.show()


# %%
def plot_timeseries(axes, x, y, color, xlabel, ylabel):
    fig, ax = plt.subplots()


# Plot the CO2 levels time-series in blue
plot_timeseries(ax, df_Pb, df_Pb["Value"], "blue", "Time (years)", "CO2 levels")

# Create a twin Axes object that shares the x-axis
ax2 = ax.twinx()

# Plot the relative temperature data in red
plot_timeseries(ax2, df_GDP.index, df_GDP["Value"], "red", "Time (years)", "Relative temperature (Celsius)")

plt.show()

# %%
