#!/usr/bin/env python
# coding: utf-8

# # IMPORTING NECESSARY LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
from   datetime import date, datetime, time
import datetime as datetime
from   statistics import mean, mode, stdev, variance
import numpy as np
import seaborn as sns
import re            # might need
import plotly.graph_objects as go
import plotly.express as px


# --------------

# # IMPORTING DATA AND CREATING DATAFRAME

df = pd.read_csv('Speed_Camera_Violations.csv')
df.head()


# ----

# # DATA CLEANING
# 

# RENAMING SOME COLUMNS THAT CONTAIN SPACES

# In[6]:


df = df.rename(columns={'CAMERA ID':'CAMERA_ID', 'VIOLATION DATE': 'V_DATE'})
df.head()


# ---

# # REMOVING UNNECESSARY COLUMNS
# 
# We will use `.pop()` function to remove unnecessary columns
# 

# In[8]:


df.pop('X COORDINATE')
df.pop('Y COORDINATE')
df.pop('LATITUDE')
df.pop('LONGITUDE')
df.pop('LOCATION')


# In[10]:


df.shape # 170521 rows and 4 columns are in `df`


# In[12]:


df.dtypes


# It looks like `ADDRESS`, `CAMERA_ID`, and `V_DATE` columns have mixed numeric and non-numeric values since their datatype is `object`.

# In[13]:


df.info()


# * Looking at the `Non-Null Count` column above, we can see that there are no null values to account for in the 4 columns of `df`.
# * Note: We could have also used `df[df.CAMERA_ID.isnull()]` to see if any rows of, for example, `CAMERA_ID` contained a null value.

# In[14]:


df.describe() # shows description of columns with only numeric data (in this case, only VIOLATIONS)


# Let's find out how many times a certain number of violations occurred 
# 
# >example: how many instances occurred where there was only 1 violation on any given day?
# 
# First, let's list the frequencies of all numbers of daily violations and analyze what we see.

# In[15]:


v_counts = pd.DataFrame(df.groupby(df['VIOLATIONS']).size())
v_counts.head()


# Let's also rename column `0` to `FREQUENCY`

# In[16]:


v_counts.rename(columns={0:'FREQUENCY'}).head()


# In[18]:


v_counts.rename(columns={0:'FREQUENCY'}).tail()


# ### **Interpretations:**
# 
# - There were 8715 instances when a camera caught just 1 violation in a day.
# - There were 2 instances when a camera caught **479** violations in a day!
# 
# 
# #### Let's further analyze those 2 instances.
# 
# 

# In[19]:


df[df.VIOLATIONS == 479]


# - **We found the corresponding camera. Its ID is CHI149. Both instances were during the Summer of 2015 at 4909 N Cicero Ave.**
# 
#     1. One possible conclusion is that this neighborhood is definitely unsafe for children since there were abnormally high number of violations in just 1 day. 
#         - One might assume that drivers tend to drive poorly in this area, and that is a serious concern. 
#         - It is possible that there is a sudden bump on the road that causes drivers to drive poorly, making them take unsafe turns or stops.
#         - Another possible reason is that there might be frequent spills of oil or other liquids on the road due to some activity either from a home on the street or a nearby factory. This could be causing drivers to lose control over their steering and thus, drive poorly.
#         - The traffic lights may not be working well. It might be the case that the lights are not bright enough for those with poorer vision and drivers might have a hard time telling the difference between green, yellow, and red.
#         - There are many possible reasons for the high number of violations here. It is important to look into the road construction, traffic lights, and surrounding activities to determine what is correlated with the high number of violations.
#         
# **However, it is also important to note that these 2 instances are from only 2 days out of all dates including in the dataset.** Therefore, it is better to plot the number of violations caught by each camera, and identify the potential outliers, and decide whether to keep them in our analysis or discard them. Outliers can serve or disrupt our analysis. It is essential to make good decisions about outliers to avoid false conclusions.
# 
# > Sidenote: Another way to filter the rows with the maximum number of accidents is to use the following code: `df[df.VIOLATIONS == max(df.VIOLATIONS)]`

# ---

# ---

# ### OVERALL AVERAGE VIOLATIONS OF EACH CAMERA

# In[20]:


v_avg_per_day = df.groupby(['CAMERA_ID', 'ADDRESS']).mean().sort_values(by='VIOLATIONS', ascending=False)
v_avg_per_day.head()


# `VIOLATIONS` does not state whether the frequency is "per day" or "per month" or "per year". Let's rename the column to make it clear.

# In[21]:


v_avg_per_day.rename(columns={'VIOLATIONS':'AVERAGE VIOLATIONS RECORDED PER DAY'})


# - The resulting table suggests that CHI149 caught more violations per day than other cameras had.
# - However, this could have also been the case due to the abnormally high number of violations on 2 days which increased the camera's overall average number of violations caught.
# 
# ---
# <br>
# 

# Let's find the total number of violations caught by each camera

# In[22]:


df.groupby('CAMERA_ID').size() # series


# The above result shows the counts of rows pertaining to each camera ID. For example, there are 1584 rows for CHI003. 
# 
# However, what we need is the number of total `VIOLATIONS` corresponding to each camera.
# 
# 

# In[53]:


sum_v = df.groupby('CAMERA_ID')[['VIOLATIONS']].sum().sort_values('VIOLATIONS')


# In[64]:


sum_v


# In[69]:


sum_v.reset_index(inplace=True)


# In[174]:


plt.figure(figsize = (50,30))
plt.xticks(rotation = 90, size = 15)
plt.yticks(size = 30)

plt.plot(sum_v.CAMERA_ID, 
         sum_v.VIOLATIONS,
         marker='o',
         markersize=6,
         linewidth=2)
plt.xlabel(xlabel = "Camera ID", 
           color = "blue", 
           size = 50)
plt.ylabel(ylabel = "Total Violations", 
           color = "blue", 
           size = 50)


# <br> I will split the data into two halves and visualize each half separately to understand the data points better. 
# 
# <br>

# In[152]:


plt.figure(figsize = (50,30))
plt.xticks(rotation = 90, size = 30)
plt.yticks(size = 30)

plt.plot(sum_v.CAMERA_ID.iloc[:int(len(sum_v)/2)], 
         sum_v.VIOLATIONS.iloc[:int(len(sum_v)/2)],
         marker='o',
         markersize=6,
         linewidth=2)
plt.xlabel(xlabel = "Camera ID", 
           color = "blue", 
           size = 50)
plt.ylabel(ylabel = "Total Violations", 
           color = "blue", 
           size = 50)


# In[169]:


plt.figure(figsize = (50,30))
plt.xticks(rotation = 90, size = 30)
plt.yticks(size = 30)

plt.plot(sum_v.CAMERA_ID.iloc[int(len(sum_v)/2): int(len(sum_v))], 
         sum_v.VIOLATIONS.iloc[int(len(sum_v)/2): int(len(sum_v))],
         marker='o',
         markersize=6,
         linewidth=2)
plt.xlabel(xlabel = "Camera ID", 
           color = "blue", 
           size = 50)
plt.ylabel(ylabel = "Total Violations", 
           color = "blue", 
           size = 50)
plt.ylim(0, 300000, 5000)


# ---

# #### notes: 
# > sum(df.groupby('CAMERA_ID')[['VIOLATIONS']].sum().sort_values('VIOLATIONS')['VIOLATIONS']) # verifies total of 4924723
# 
# 
#        (81)   0 - 81                 |          (81)    81 - 161
# 
#        0 - len(sum_v.CAMERA_ID/2)    |          len(sum_v.CAMERA_ID/2) - len(sum_v.CAMERA_ID)-1
#        
#   - df.groupby('CAMERA_ID').size().count() # quicker way to find number of cameras
#   - df.groupby(['CAMERA_ID', 'ADDRESS']).size()
#   - str(df.groupby(['CAMERA_ID', 'ADDRESS']).size().count()) + ' unique addresses' # to output result with details

# ---

# ----

# 

# **If we want to further analyze the number of days that any camera had caught any number of violations, we can run by the following code (this will display only the first 20 rows):** 
# 
# <br>

# In[186]:


df.head()


# In[203]:


# pd.DataFrame(df.groupby(['VIOLATIONS', 'CAMERA_ID']).size()).rename(columns={0:'COUNT_OF_DAYS'}).sort_index(ascending=True)[:20]


# In[208]:


df2 = pd.DataFrame(df.groupby(['VIOLATIONS', 'CAMERA_ID']).size()).rename(columns={0:'COUNT_OF_DAYS'}).sort_index(ascending=True).sort_values(['VIOLATIONS', 'COUNT_OF_DAYS', 'CAMERA_ID'])


# In[211]:


df2.reset_index(inplace=True)


# In[223]:


sns.scatterplot(y="VIOLATIONS", x="CAMERA_ID", hue="CAMERA_ID", size="COUNT_OF_DAYS", data=df2[100:110])


# **INTERPRETATION:** Looking at the first row of the dataframe above, we can see that there were 15 days on which the camera CHI005 caught just 1 violation. 
# 
# <br>
# 
# Let's order the rows in descending order of the `VIOLATIONS`.

# In[17]:


numberOfDaysOfViolations = pd.DataFrame(df.groupby(['VIOLATIONS', 'CAMERA_ID']).size().sort_index(ascending=False))[0:147].rename(columns={0:'Count of Days on which `CAMERA_ID` caught `VIOLATIONS` number of violations'})
numberOfDaysOfViolations.head()


# We can see that camera CHI149 caught 479 violations on 2 days.

# ---
# 
# <br><br>

# Let's further analyze the violations caught by camera CHI149.

# In[17]:


df[df['CAMERA_ID']=='CHI149']


# In[187]:


df[df['CAMERA_ID']=='CHI149']['VIOLATIONS'].sum()


# Total number of violations caught by CHI149 = 296,755

# In[188]:


df[['CAMERA_ID','VIOLATIONS']]['VIOLATIONS'].sum()


# Total number of violations caught by all cameras = 4,924,723 

# ### Interpretation: 
# 
# - Camera CHI149 caught 296,755 violations out of all the 4,924,723 violations in total over the years.
# - CHI149 caught (296755/4924723*100)% of all violations = 6.026% of all violations.
# 

# Even though CHI149 caught so many violations in 2 days, it only recorded 6.026% of all the violations.
# 
# Let's analyze what other cameras caught large number of violations to better understand what is going on.

# ---

# ---

# But first, let's analyze CHI149 violation patterns over time

# In[189]:


chi149 = df[df['CAMERA_ID']=='CHI149'].sort_values('V_DATE')


# In[190]:


chi149.head()


# In[191]:


chi149.count() # chi149 contains 1506 rows


# In[192]:


chi149Dates = chi149[['V_DATE', 'VIOLATIONS']]
chi149Dates.sort_values(by=['V_DATE'])


# - The dates are not in order. 
# - We will fix this error by splitting the dates in the original dataframe, `df`, into month, day, and year, and then order the dates based on year, and then month, and finally, day.

# In[193]:


# Already imported datetime as datetime 
# to extract months, dates, and years of dates 
# in order to find the earliest & latest date available.

# creating new columns
df['month'] = pd.DatetimeIndex(df['V_DATE']).month
df['date'] = pd.DatetimeIndex(df['V_DATE']).day
df['year'] = pd.DatetimeIndex(df['V_DATE']).year

sortedDates = df.sort_values(['year', 'month', 'date'])
sortedDates


# **Important Note:** We could have used `df.sort_values('V_DATE')` but it gives the wrong order of dates. The corresponding output will say 01/01/2015 is the earliest (it is the first row of the output), 
# and 12/31/2017 is the latest. The reason is that the automatic sorting order is month, date, and then year. However, we need the dates ordered by year, month, and finally, date. 
# 
# We can do so using the following code:
# 
# > `df.sort_values(['year', 'month', 'date'])` 
# 
# as we did above.
# 
# After ordering the dates correctly, we can see that the earliest date of the whole dataset is 07/01/2014 and the latest is 12/23/2018.
# 
# <br>
# 
# 

# <br>
# Let's consider only the rows related to CHI149 again.

# In[194]:


chi149 = sortedDates[sortedDates['CAMERA_ID']=='CHI149'] # since already sorted, do not require `.sort_values(by=['year', 'month', 'date'])`
chi149


# #### We can see that 11/01/2014 is the earliest and 12/23/2018 is the latest recorded date for CHI149.
# 
# <br><br>
# 
# Now that we have ordered all the rows of chi149, let's plot the dates and number of violations. There are a total of 1506 rows, so the resulting plot will be messy since each date appears only once in this subset of data, and there are many dates to be displayed on the x-axis. So, the x-axis is very hard to read.

# In[60]:


plt.plot_date(chi149.V_DATE, chi149.VIOLATIONS)


# Let's change the plot size and maybe even create subplots for each year.

# In[63]:


plt.figure(figsize=(20,20))
plt.plot_date(chi149.V_DATE, chi149.VIOLATIONS)


# ---

# To make the plot look neater, let's use seaborn (aliased as `sns`)

# In[156]:


sns.set(rc={'figure.figsize':(60,30)})
sns.swarmplot(data=chi149,
            x="year",
            y="month",
            hue="VIOLATIONS",
            size=10,  
            palette="rainbow")


# - In 2015, there are many dates where the total number of violations is very high (red dots).
# - Hence, in the year 2015, CHI149 caught the most number of violations per year.
# 

# In[195]:


sns.set(rc={'figure.figsize':(30,60)})
sns.swarmplot(data=df[300:1000],
            x="year",
            y="month",
            hue="VIOLATIONS",
            size=10,
            alpha=0.5,  
            palette="rainbow")


# In[ ]:


df.head()


# In[ ]:


fig = px.scatter(chi149, 
                 x="year", 
                 y="month", 
                 color="VIOLATIONS",
                 color_continuous_scale='Inferno',
                 hover_name="VIOLATIONS",
                 size=chi149.VIOLATIONS//10,
                 animation_frame='V_DATE')
fig.show()


# ----

# In[277]:


a = chi149.sort_values('VIOLATIONS').groupby(chi149.VIOLATIONS).size()
a


# In[295]:


b = chi149.sort_values('V_DATE').groupby(['year','month','date']).size()
b = pd.DataFrame(b)
b.index


# In[306]:


df.sort_values(['year', 'month', 'date', 'VIOLATIONS', 'CAMERA_ID', 'ADDRESS'])


# In[307]:


ymdVCA = df.sort_values(['year', 'month', 'date', 'VIOLATIONS', 'CAMERA_ID', 'ADDRESS'])


# In[308]:


type(ymdVCA)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Without using `legend="full"`, the legend only showed years 2014, 2015, 2017, and 2018. 
# 
# However, there are records from 2016, so we must use `legend="full"`, as done above, to include 2016 data as well.

# In[157]:



sns.set(rc={'figure.figsize':(30,20)})
sns.catplot(data=chi149, x="V_DATE", y="VIOLATIONS", hue="month", palette="rainbow", legend=True)


# ---

# In[309]:


sns.color_palette("husl", 8)

ax = sns.barplot(x="year", 
                 y="VIOLATIONS", 
                 hue="CAMERA_ID", 
                 data=chi149, 
                 errcolor="lightgray",
                 errwidth=1
                )
ax


# We can see that the number of violations was indeed the highest during months 6-8 (June-August) of 2015.
# 
# We can also see that during each year, there is a peak in violations during the Summer months and then a reduction during the Fall/Winter months.
# 
# It is possible that the neighborhood is extra crowded during the Summer and less during the Winter.
# 
# It could be the case that the Summer heat is correlated to poorer focus of the drivers, leading to more violations.
# 
# It is important to look into the area and observe changes.

# ---

# In[ ]:





# In[150]:


chi149.V_DATE #[0:((chi149.V_DATE.count())/2)] / returns decimal. but cant use decimal for slicing index.


# In[75]:


chi149.V_DATE[0:(chi149.V_DATE.count()//2)] # 753 rows


# In[76]:


10/2  # 5.0


# In[77]:


10//2 # 5


# In[52]:


chi149.V_DATE.count()


# In[106]:


chi149.V_DATE[0:1506]


# In[53]:


chi149.VIOLATIONS.count()


# In[56]:


chi149.VIOLATIONS[1506:]


# In[50]:


chi149.V_DATE[0:chi149.V_DATE.count(),]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[107]:


chi149.VIOLATIONS.count()


# In[108]:


chi149.VIOLATIONS[chi149.VIOLATIONS.count()]


# In[ ]:





# In[ ]:





# In[22]:


df['CAMERA_ID'].value_counts().sort_index()


# 162 diff cameras in total (shown by Length: 162 below the series above.)

# In[ ]:





# In[ ]:





# In[23]:


len(df.V_DATE.value_counts())     # 1637
len(df.ADDRESS.value_counts())    # 163
len(df.CAMERA_ID.value_counts())  # 162


# In[24]:




min(df.V_DATE)                    
# ?? '01/01/2015' but actually it is 7/1/2014 .. wrong order because month is taken 
# ?? into consideration first, followed by date, then year.

max(df.V_DATE)                    
# '12/31/2017'




# In[ ]:





# In[25]:


address = pd.DataFrame(df.groupby('ADDRESS').size().sort_values(ascending=False))
address.rename(columns={0:'FREQUENCY AT GIVEN ADDRESS'})


# In[ ]:





# In[26]:


pd.DataFrame(df.groupby(['VIOLATIONS', 'CAMERA_ID']).size().sort_index(ascending=False))[0:20].rename(columns={0:'Count of Days on which `CAMERA_ID` caught `VIOLATIONS` number of violations'})


# In[ ]:





# In[27]:


pd.DataFrame(df.groupby(['VIOLATIONS', 'CAMERA_ID']).size()[0:20]).rename(columns={0:'Count of Days on which `CAMERA_ID` caught `VIOLATIONS` number of violations'})


# In[28]:


pd.DataFrame(df.groupby('year').size().sort_values(ascending=False)).sort_index()


# Note: numbers are just number of rows that contain year value of 2014, not the actual number of accidents in 2014. This is further confirmed below.

# In[29]:


df_2014 = df[df.year == 2014]
df_2014


# In[30]:


min(df_2014.month)


# In[31]:


df_7_2014 = df_2014[df_2014.month==7]


# In[32]:


df_7_2014.sort_values(by='date')


# Shortcut:

# In[33]:


df_2014.sort_values(by=['month', 'date'])


# **Proof that 07-01-2014 is the earliest date and not 1-1-2015**
# 
# ---

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


violations = pd.DataFrame(df.groupby('VIOLATIONS').size())
violations.columns=['Number of Occurences of X Violations']
violations


# In[35]:


df['VIOLATIONS'].value_counts().sort_index()[-10:]


# Confirmation: There were 2 instances of **479 violations in just 1 day!**

# In[36]:


df[df.VIOLATIONS==479]


# Interestingly, both those days involved the same `CAMERA_ID` and `ADDRESS`, showing us that this area is a very unsafe area.

# In[37]:


v60 = pd.DataFrame(df[df.VIOLATIONS>60])
v60


# In[38]:


v60_1 = pd.DataFrame(v60.groupby('CAMERA_ID').size().sort_values(ascending=False))
v60_1

df[df.VIOLATIONS>300].groupby('CAMERA_ID').size().sort_values(ascending=False)     # number of ROWS, not violations


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


# df.loc[df['CAMERA_ID']=='CHI069'].groupby(['ADDRESS']).size() # 1632
# df.loc[df['CAMERA_ID']=='CHI045'].groupby(['ADDRESS']).size() # 1620
# df.loc[df['CAMERA_ID']=='CHI063'].groupby(['ADDRESS']).size() # 821
df.loc[df['CAMERA_ID']=='CHI149'].groupby(['ADDRESS']).size() # 1506


# ---

# trick to find months, days, years:

# In[40]:


monthOfAcc = []
for date in df.V_DATE[0:20]:
    month = date[0:2]
    print(month)
    monthOfAcc.append(month)


# In[41]:


dayOfAcc = []
for date in df.V_DATE[0:20]:
    day = date[3:5]
    print(day)
    dayOfAcc.append(day)


# In[42]:


yearOfAcc = []
for date in df.V_DATE[0:20]:
    year = date[-4:]
    print(year)
    yearOfAcc.append(year)


# In[ ]:





# ----

# In[43]:


# df['monthOfAcc'] = monthOfAcc
# df['dayOfAcc'] = dayOfAcc
# df['yearOfAcc'] = yearOfAcc


# In[ ]:





# ----

# In[44]:


df.head()


# In[45]:


year_month = df.groupby(['year', 'month']).size()
plt.figure(figsize=(20,5))
year_month.plot(kind='bar', color='pink')


# The grouping shows us that the earliest date recorded was 07/2014 and latest was 12/2018.
# 
# There seems to be a pattern over the years where the accidents in any given year seem to be the lowest during the month of August compared to other months in the same year.
# 
# Yearly peaks also seem to be during March and May. It is possible that Spring break or Summer break might have something to do with it, since more people are likely to travel.

# In[46]:


year_month = df.groupby(['year', 'month']).size().sort_values(ascending=True)
plt.figure(figsize=(20,5))
year_month.plot(kind='bar', color='pink')


# There were more accidents in October of 2018 than during other times. Maybe Halloween of 2018 was very crowded or simply wilder than during other times.

# In[47]:


df.groupby('year').mean()['VIOLATIONS']


# per day, 2014 shows more. But probably because focussing on proportion of second half of year, the busier time of the year.

# 

# it looks like 2014 had significantly less violations in total, but we must remember for year 2014, we only have data from July 1st, 2014 as shown below.

# ----

# In[48]:


plt.figure(figsize=(15,6), facecolor='lightpink', frameon=True)
plt.hist(sorted(df.month), color='lightgreen', width=0.5)
plt.xlabel('MONTH')
plt.ylabel('FREQUENCY')
# plt.xlim(range(0,13))  # includes 12
# plt.xticks(range(1,14), ['j','f','m','a','m','j','j','a','s','o','n','d']) # doesn't include last number, 13
plt.show()


# In[49]:


df.sort_values(['year', 'month', 'date'])[0:10]


# In[ ]:





# In[ ]:





# In[ ]:





#  
#  
#  
#  
#  
#  

#   ---

# In[50]:


df['V_DATE'].value_counts().sort_values()


# On 02/01/2015, there were a total of only 16 rows/records in the dataset!
# 
# However, on 11/30/2018 (around Thanksgiving time), 151 records were recorded in total for all the neighborhoods in the dataset.

# In[ ]:





# # **below is all about the number of records - don't get tricked into thinking it's about number of violations**

# Note:
# 
# To get a  better idea of how years 2015-2018 compare, let's exclude 2014 data

# In[51]:


plt.plot((df['year'].value_counts().sort_index())[1:])


#  and zoom in such that the yaxis limits only start from around 37000.

# In[52]:


plt.plot((df['year'].value_counts().sort_index())[1:])
plt.yticks(range(30000, 45000, 5000))


# Zooming out a little, we can see that the change is present, but remains very high throughout the years. Zooming in enabled us to see exactly what the differences are. Let's zoom in even more

# In[53]:


plt.plot((df['year'].value_counts().sort_index())[1:])
plt.yticks(range(37800, 39200, 100))
plt.show()


# Zooming in as much as possible shows us that hundreds of more violations occurred in 2016 than the other years! 
# 
# 2015 to 2016 : around 1000 more violations occurred
# 
# 2016 to 2017 : violations reduced by about 600 violations
# 
# 2017 to 2018 : violations reduced by about 100-200 violations 

# In[54]:


df['year'].value_counts().sort_values()


# As the graph shows, 2016 is indeed the most frequent year in the dataset. 

#  

# -------

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[55]:


address_values = df.groupby('ADDRESS').size().sort_values(ascending= False)[0:10]
plt.figure(figsize=(500,200))

plt.subplot(211)
plt.plot(address_values, marker="o", color='green')
plt.xticks(rotation=90)
plt.show()

