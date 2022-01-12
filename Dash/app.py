import pandas as pd
import numpy as np
import dash
from dash import html
from dash import dcc
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# load our pre-cleaned data in. This loads the data from our earlier datacleaning and analysis.
df = pd.read_csv('../cleaned_train_df.csv')
pred_res_df = pd.read_csv('../pred_res.csv')

# get the columns of interest, put them in an array. Map their names to more understandable ones using a dictionary
df['company_size']=df['company_size'].astype(str).str.replace('/','-')
df['target']=df['target'].astype(str).str.replace('0','No').replace('1','Yes')
columns1 = ["gender","education_level","major_discipline","enrolled_university","relevent_experience","experience","company_size","company_type","last_new_job"]
column1_dict = {"gender":"Gender","education_level":"Education Level","major_discipline":"Major Discipline","enrolled_university":"University Enrolment","relevent_experience":"Years of Relevant Experience (in Data Science)","experience":"Total Years of Experience","company_size":"Number of Employees in current company","company_type":"Type of current company","last_new_job":"Years since last job change"}

# translate pred_res_df column labels into user friendly titles
model_dict = {"prediction_lr":"Logistic Regression Model","prediction_rf":"Random Forests Model", "prediction_knn":"K-Nearest Neighbors Model"}
model_col = ["prediction_lr","prediction_rf","prediction_knn"]

# Initialise the app
app = dash.Dash(__name__)

#Callback for the first interactive graph - countplot
@app.callback( Output('countplot', 'figure'),
              [Input('selector1', 'value')])
def update_plot(selector1):
    # get the selected value
    selected = column1_dict.get(selector1)
    # plot the bar graph
    figure = px.histogram(df, x=selector1, color="target", barmode='group',labels={selector1:selected,"target":"Looking for a job change"},title="Countplot for how desire to look for a new job changes with "+str(selected))
    # set title for the legend
    figure.update_layout(legend_title_text='Looking for a job change')
   #Return the figure we produce so the HTML can be updated with it.
    return figure

#Callback for the second interactive graph - boxplot
@app.callback( Output('boxplot', 'figure'),
              [Input('selector2', 'value')])
def update_plot(selector2):
    # get the selected value
    selected = column1_dict.get(selector2)
    # draw boxplot for the selected column
    figure = px.box(df, x=selector2, y="training_hours", color="target",labels={selector2:selected,"training_hours":"Training hours","target":"Looking for a job change"},title="Boxplot breakdown of Training hours by "+str(selected)+" and whether they affect job-changing decisions")
    figure.update_layout(legend_title_text='Looking for a job change')
   #Return the figure we produce so the HTML can be updated with it.
    return figure

#Callback for the third interactive graph - boxplot
@app.callback( Output('modelplot', 'figure'),
              [Input('selector3', 'value')])
def update_plot(selector3):
    # Get user friendly title of selected model
    selected = model_dict.get(selector3)

    # Get array from pred_res_df based on user-selected model
    model_arr = pred_res_df[selector3].to_numpy()

    result_arr= pred_res_df['real_result'].to_numpy()

    # Create confusion matrix figure
    figure = px.imshow([model_arr, result_arr], aspect="auto", y=[selected + " Result", "Actual Result"], 
        labels=dict(x="Number of Entries", color = "Looking for a job change"), title=selected + " Confusion Matrix (0 - Not looking for a job change, 1 - Looking for a job change)")
    # Return the figure we produce so the HTML can be updated with it.
    return figure


# Define the app layout
app.layout = html.Div(
    children=[
        html.Div(className='row',  # Define the row element
                 children=[
                     # Define the left element
                     html.Div(className='four columns div-user-controls',
                              children = [
                                  html.H1('Job change indicator Visualizer'),
                                  html.H2('1.Variables affecting Job change decisions'),
                                  html.P('''The first plot visualizes how the number of people looking for job changes can vary between different groups.'''),
                                  html.P('''Recruiters can look at this graph to have a general idea of the profile of people who are taking training because they want a job change.'''),
                                  html.P('''This can help them narrow down possible candidates more easily and make recruitment more efficient.'''),
                                  # Adding option to select columns -- this is the first column list
                                  html.Div(className='div-for-dropdown',
                                           children=[
                                               dcc.Dropdown(id='selector1',
                                                            options=[
                                                                {"label": column1_dict.get(i), "value": i}
                                                                for i in columns1
                                                            ],
                                                            multi=False,
                                                            placeholder="Select a column",
                                                            
                                                           )
                                           ]
                                          ),html.Br(),html.H2('2.Training hours and Job change decisions'),
                                  html.P('''The second plot visualizes the training hours vary depending on the group and whether they want to change jobs.'''),
                                  html.P('''Again, this graph can help recruiters find possible candidates by combining different variables and looking for those within a specific training hours range.'''),
                                  # Adding option to select columns -- this is the second column list
                                  html.Div(className='div-for-dropdown',
                                           children=[
                                               dcc.Dropdown(id='selector2',
                                                            options=[
                                                                {"label": column1_dict.get(i), "value": i}
                                                                for i in columns1
                                                            ],
                                                            multi=False,
                                                            placeholder="Select a column",
                                                            
                                                           )
                                           ]
                                          ),html.Br(),html.H2('3.Model confusion matrix plot'),
                                  html.P('''A heatmap-based confusion matrix plot for all three models built by our team.'''),
                                  html.P('''Each option in the dropdown represents one model. The plot shows the model predicted result (above) compared to the actual result (below - taken from the dataset). If the prediction matches the actual result, they will have the same color.'''),
                                  # Adding option to select models -- this is the model list
                                  html.Div(className='div-for-dropdown',
                                           children=[
                                               dcc.Dropdown(id='selector3',
                                                            options=[
                                                                {"label": model_dict.get(i), "value": i}
                                                                for i in model_col
                                                            ],
                                                            multi=False,
                                                            placeholder="Select a model",
                                                            
                                                           )
                                           ]
                                          ),
                              ]
                             ),html.Div(className='eight columns div-for-charts bg-grey',
                              children = [
    dcc.Graph(
        #Countplot here, the first graph
        id='countplot'
    ),dcc.Graph(
        #And the second graph.
        id='boxplot'
    ),dcc.Graph(
        #And the third graph.
        id='modelplot'
    )
                     
                 ]
                ),])])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False, dev_tools_props_check=True)

