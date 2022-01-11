import pandas as pd
import numpy as np
import dash
from dash import html
from dash import dcc
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# load our pre-cleaned data in. This loads the data from our earlier datacleaning and analysis.
df = pd.read_csv('../cleaned.csv', dtype='category')
# We're going to be interested in annual salaries, education, and coding experience again. However, this time we're going to add in country of residence.
df["ConvertedCompYearly"] = pd.to_numeric(df["ConvertedCompYearly"], errors='raise')
#Move all the data we're interested in for our first graph into a subset
selected = df[["Country","ConvertedCompYearly","YearsCode","EdLevel"]]
df_subset = selected.copy()
#We're going to need some slightly different data for a second graph
selected = df[["Country","EdLevel"]]
df_edu = selected.copy()
#Salaries have a ruge range, so we do a logarithmic transformation again, so we can see things more clearly
df_subset['logsal'] = np.log10(df_subset['ConvertedCompYearly'])
#We repeat the same modification again to categorize years of coding experience into 5 year periods. This is much better for visualization. For our first graph, we also need to drop missing values.
df_subset = df_subset.drop(df_subset.index[df_subset['YearsCode'].isin(["More than 50 years"])])
df_subset = df_subset.dropna()
df_subset['YearsCode'] = df_subset['YearsCode'].cat.add_categories('0')
df_subset.loc[(df_subset['YearsCode'].isin(["Less than 1 year"])),['YearsCode']] = "0"
df_subset["YearsCode"] = pd.to_numeric(df_subset["YearsCode"], errors='raise')
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
names = ['1-5yrs', '6-10yrs', '11-15yrs', '16-20yrs', '21-25yrs', '26-30yrs', '31-35yrs','36-40yrs','41-45yrs','46-50yrs']
df_subset['yrsrange'] = pd.cut(df_subset['YearsCode'], bins, labels=names)
#For our second graph, we will simplify the education levels as we did in our earlier analysis. We'll be using count data -- so we can either drop NaN values or set them to zero -- same result.
df_edu['SimpleEdLevel'] = df_subset['EdLevel'].replace({'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 'High School', 'Some college/university study without earning a degree': 'Some Uni', 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 'BDeg', 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 'MDeg','Other doctoral degree (Ph.D., Ed.D., etc.)':'PhD'})
df_edu.loc[(df_edu['SimpleEdLevel'].isin(["Primary/elementary school","Associate degree (A.A., A.S., etc.)","Professional degree (JD, MD, etc.)","Something else"])),['SimpleEdLevel']] = np.nan
df_edu = df_edu.dropna()

# Create a list of all countries that have real values in the first dataset.
country_list = []
x = (df_subset['Country'].unique())
for i in x:
    country_list.append(i)  

country_list = sorted(country_list)

# Do the same for the second dataset, the missing values can be different.
country_list2 = []
x = (df_edu['Country'].unique())
for i in x:
    country_list2.append(i)  

country_list2 = sorted(country_list2)

# Initialise the app
app = dash.Dash(__name__)

#Callback for the first interactive graph. We originally wanted to do one selector for both graphs, but it was very buggy, and then I realized that the country list is slightly different for each due to the distribution of missing values.
@app.callback( Output('scatterplot', 'figure'),
              [Input('selector1', 'value')])
def update_plot(selector1):
    #The tutorial code had a bug where removing countries sometiems didn't result in updates. Fixed it by resetting the list of active countries in a few spots.
    countries = []
    print(selector1)
    if type(selector1) == list:
        #Try / except deals with an edge case where a user can pass an empty or invalid list to the application.
        try:
            for country in selector1:
                print(country)
                countries.append(country)
        except:
            countries = []
        #We take a list of all selected countries, create a dataframe containing data from all of them, and add each to the boxplot as a different color. This lets us compare different countries visually which is a neat data exploration tool. 
        df_selected_country = df_subset.loc[df_subset['Country'].isin(countries)]
        #plotly express has a hard time sorting categorical data on the x axis -- it appears in the order it is in the data. So we sort the data alphabetically by our category data, which happens to display exactly
        #the way we want it when sorted alphabetically. The longer alternative was sorting the data in a custom order.
        df_selected_country = df_selected_country.sort_values(by="yrsrange")
        figure = px.box(df_selected_country, x="yrsrange", y="logsal",color="Country",title="Logarithm of Salary vs. Years of Coding Experience in Various Countries",labels={'yrsrange':'Years of Coding Experience', 'logsal':'Salary (base 10 logarithmic scale)'})
    #Same as above.
    else:
        try:
            for country in selector1:
                countries.append(country)
        except:
            countries = []
        df_selected_country = df_subset.loc[df_subset['Country'].isin(countries)]
        df_selected_country = df_selected_country.sort_values(by="yrsrange")
        figure = px.box(df_selected_country, x="yrsrange", y="logsal",color="Country",title="Logarithm of Salary vs. Years of Coding Experience in Various Countries",labels={'yrsrange':'Years of Coding Experience', 'logsal':'Salary (base 10 logarithmic scale)'})
   #Return the figure we produce so the HTML can be updated with it.
    return figure



#Callback for the second graph, an interactive bargraph
@app.callback( Output('bargraph', 'figure'),
              [Input('selector2', 'value')])
def update_plot(selector2):
    countries = []
    edulist = []
    print(selector2)
    #The way this works is slightly horrible. We decided on a 'grouped bar graph', which is created by iterating through a list of countries the user has selected, creating a bar graph for each and appending to a list of bar graphs.
    #Then we concatenate them into a plotly graph object and set barmode=group. 
    if type(selector2) == list:
        try:
            j=0
            for country in selector2:
                j=j+1
                temp_df = df_edu.loc[df_edu['Country'] == country]
                #The lines below are a bit of an ugly hack. We set the index to our education levels, then replace all other columns with the count data for those education levels. Either of those other columns are used as the useful output.
                temp_df=temp_df.groupby('SimpleEdLevel').count().reset_index()
                #Calculate a total count
                total = temp_df.sum()["EdLevel"]
                #Convert counts to a proportion -- this makes it much easier to compare two countries with different amounts of survey respondants.
                temp_df['EdLevel'] = temp_df['EdLevel'].div(total).round(2)
                #Create a bar grap hand append to our list of bar graphs
                edulist.append(go.Bar(name=country, x=temp_df['SimpleEdLevel'], y=temp_df["EdLevel"], offsetgroup=j))
        except:
            countries = []
        #Make a grouped bar graph using our list of bar graphs.
        figure2 = go.Figure(data=edulist,layout={'title': 'Proportion of Coders who have Obtained Different Degrees in Selected Countries',
        'yaxis': {'title': 'Proportion of Coders'},'xaxis': {'title': 'Degree Obtained'}})
        #Group the bar graphs
        figure2.update_layout(barmode='group')
        #Reorder the x axis. TWe don't consider them ordinal categories so this isn't strictly speaking necessary, but I think it's easier to read this way.
        figure2.update_xaxes(categoryorder='array', categoryarray=['High School','Some Uni','BDeg','MDeg','PhD'])
        
    #Same as above
    else:
        try:
            j=0
            for country in selector2:
                j=j+1
                temp_df = df_edu.loc[df_edu['Country'] == country]
                temp_df=temp_df.groupby('SimpleEdLevel').count().reset_index()
                total = temp_df.sum()["EdLevel"]
                temp_df['EdLevel'] = temp_df['EdLevel'].div(total).round(2)
                edulist.append(go.Bar(name=country, x=temp_df['SimpleEdLevel'], y=temp_df["EdLevel"], offsetgroup=j))
        except:
            countries = []
        figure2 = go.Figure(data=edulist,layout={'title': 'Proportion of Coders who have Obtained Different Degrees in Selected Countries',
        'yaxis': {'title': 'Proportion of Coders'},'xaxis': {'title': 'Degree Obtained'}})
        figure2.update_layout(barmode='group')
        figure2.update_xaxes(categoryorder='array', categoryarray=['High School','Some Uni','BDeg','MDeg','PhD'])
    #Return the figure we produce so the HTML can be updated with it.
    return figure2




# Define the app layout. We re-use a lot from Tutorial 3.
app.layout = html.Div(
    children=[
        html.Div(className='row',  # Define the row element
                 children=[
                     # Define the left element
                     html.Div(className='four columns div-user-controls',
                              children = [
                                  html.H1('Programming Career Visualizer'),
                                  html.H2('1.Salary and Experience'),
                                  html.P('''We visualize how salary increases (or doesn't) over time in different countries.'''),
                                  html.P('''Pick one or more countries to plan if and when you should emigrate for a better career. Remember salaries are on a logarithmic scale.'''),
                                  html.P('''Or, you know... use it to figure out where you can outsource work most cheaply to experienced programmers, and stay exactly where you are.'''),
                                  # Adding option to select columns -- this is the first country list
                                  html.Div(className='div-for-dropdown',
                                           children=[
                                               dcc.Dropdown(id='selector1',
                                                            options=[
                                                                {"label": i, "value": i}
                                                                for i in country_list
                                                            ],
                                                            multi=True,
                                                            placeholder="Select a country",
                                                            
                                                           )
                                           ]
                                          ),html.H2('2.Education Levels in the Coding Job Market'),
                                  html.P('''We visualize what the educational degree breakdown of different countries are too.'''),
                                  html.P('''If you're going to emigrate, better to know what the competition in the job market is like!'''),
                                  html.P('''Also, if you're doing outsourcing, you might want to know what level of education coders typically have in a country.'''),
                                  # Adding option to select columns -- this is the second country list
                                  html.Div(className='div-for-dropdown',
                                           children=[
                                               dcc.Dropdown(id='selector2',
                                                            options=[
                                                                {"label": i, "value": i}
                                                                for i in country_list2
                                                            ],
                                                            multi=True,
                                                            placeholder="Select a country",
                                                            
                                                           )
                                           ]
                                          ),
                              ]
                             ),html.Div(className='eight columns div-for-charts bg-grey',
                              children = [
    dcc.Graph(
        #Scatterplot here, the first graph
        id='scatterplot'
    ),dcc.Graph(
        #And the second graph.
        id='bargraph'
    )
                     
                 ]
                ),])])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=True, dev_tools_props_check=True)

