from scipy import interpolate
import dash_table
import mydcc
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.integrate import odeint


############################################ the model ################################################


def deriv(y, t, r0_y_interpolated, gamma, sigma, N, p_I_to_C, p_C_to_D, Beds):
    S, E, I, C, R, D = y

    def betaa(t):
        return I / (I + C) * (12 * p_I_to_C + 1/gamma * (1 - p_I_to_C)) + C / (I + C) * (
            min(Beds(t), C) / (min(Beds(t), C) + max(0, C-Beds(t))) * (p_C_to_D * 7.5 + (1 - p_C_to_D) * 6.5) +
            max(0, C-Beds(t)) / (min(Beds(t), C) + max(0, C-Beds(t))) * 1 * 1
        )

    def beta(t):
        try:
            return r0_y_interpolated[int(t)] / betaa(t) if not np.isnan(betaa(t)) else 0
        except:
            return r0_y_interpolated[-1] / betaa(t)
            
    dSdt = -beta(t) * I * S / N
    dEdt = beta(t) * I * S / N - sigma * E
    dIdt = sigma * E - 1/12.0 * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    dCdt = 1/12.0 * p_I_to_C * I - 1/7.5 * p_C_to_D * \
        min(Beds(t), C) - max(0, C-Beds(t)) - \
        (1 - p_C_to_D) * 1/6.5 * min(Beds(t), C)
    dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * \
        1/6.5 * min(Beds(t), C)
    dDdt = 1/7.5 * p_C_to_D * min(Beds(t), C) + max(0, C-Beds(t))
    return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt



gamma = 1.0/9.0
sigma = 1.0/3.0

def logistic_R_0(t, R_0_start, k, x0, R_0_end):
    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end

def Model(initial_cases, initial_date, N, beds_per_100k, R_0_start, k, x0, R_0_end, p_I_to_C, p_C_to_D, s, r0_y_interpolated=None):
    days = 360
    def beta(t):
        return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma
    
    def Beds(t):
        beds_0 = beds_per_100k / 100_000 * N
        return beds_0 + s*beds_0*t  # 0.003


    diff = int((np.datetime64("2020-01-01") - np.datetime64(initial_date)) / np.timedelta64(1, "D"))
    if diff > 0:
        r0_y_interpolated = [r0_y_interpolated[0] for _ in range(diff-1)] + r0_y_interpolated
    elif diff < 0:
        r0_y_interpolated = r0_y_interpolated[(-diff):]

    last_date = np.datetime64(initial_date) + np.timedelta64(days-1, "D")
    missing_days_r0 = int((last_date - np.datetime64("2020-09-01")) / np.timedelta64(1, "D"))
    r0_y_interpolated += [r0_y_interpolated[-1] for _ in range(missing_days_r0+1)]

    y0 = N-initial_cases, initial_cases, 0.0, 0.0, 0.0, 0.0
    t = np.linspace(0, days, days)
    print(t)
    ret = odeint(deriv, y0, t, args=(r0_y_interpolated,
                                        gamma, sigma, N, p_I_to_C, p_C_to_D, Beds))
    S, E, I, C, R, D = ret.T
    R_0_over_time = r0_y_interpolated
    total_CFR = [0] + [100 * D[i] / sum(sigma*E[:i]) if sum(
        sigma*E[:i]) > 0 else 0 for i in range(1, len(t))]
    daily_CFR = [0] + [100 * ((D[i]-D[i-1]) / ((R[i]-R[i-1]) + (D[i]-D[i-1]))) if max(
        (R[i]-R[i-1]), (D[i]-D[i-1])) > 10 else 0 for i in range(1, len(t))]



    dates = pd.date_range(start=np.datetime64(initial_date), periods=days, freq="D")

    return dates, S, E, I, C, R, D, R_0_over_time, total_CFR, daily_CFR, [Beds(i) for i in range(len(t))]


############################################ the dash app layout ################################################
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "CoSim"


# these are the controls where the parameters can be tuned.
# They are not placed on the screen here, we just define them.
# Each separate input (e.g. a slider for the fatality rate) is placed
# in its own "dbc.FormGroup" and gets a "dbc.Label" where we put its name.
# The sliders use the predefined "dcc.Slider"-class, the numeric inputs
# use "dbc.Input", etc., so we don't have to tweak anything ourselves.
# The controls are wrappen in a "dbc.Card" so they look nice.
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Date of first infection'),
                html.Br(),
                dcc.DatePickerSingle(
                    day_size=39,  # how big the date picker appears
                    display_format="DD.MM.YYYY",
                    date='2020-01-01',
                    id='initial_date',
                    min_date_allowed=dt(2020, 12, 1),
                    max_date_allowed=dt(2020, 5, 31),
                    initial_visible_month=dt(2020, 1, 15),
                    placeholder="test"
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Initial Cases"),
                dbc.Input(
                    id="initial_cases", type="number", placeholder="initial_cases",
                    min=1, max=1_000_000, step=1, value=10,
                )
            ]
        ),

        dbc.FormGroup(
            [
                dbc.Label("Population"),
                dbc.Input(
                    id="population", type="number", placeholder="population",
                    min=10_000, max=1_000_000_000, step=10_000, value=80_000_000,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('ICU beds per 100k people'),
                dbc.Input(
                    id="icu_beds", type="number", placeholder="ICU Beds per 100k",
                    min=0.0, max=100.0, step=0.1, value=34.0,
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Probability of going to ICU when infected (%)'),
                html.Br(),
                dcc.Slider(
                    id='p_I_to_C',
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    value=5.0,
                    tooltip={'always_visible': True, "placement": "bottom"}
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Probability of dying in ICU (%)'),
                dcc.Slider(
                    id='p_C_to_D',
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    value=50.0,
                    tooltip={'always_visible': True, "placement": "bottom"}
                ),
            ]
        ),
        # this is the input where the R value can be changed over time.
        # It is implemented as a table where the date is in the first column,
        # and users can change the R value on that date in the second column.
        dbc.FormGroup(
            [
                dbc.Label('Reproduction rate R over time'),
                dash_table.DataTable(
                    id='r0_table',
                    columns=[
                        {"name": "Date", "id": "Date"},
                        {"name": "R value", "id": "R value",
                         "editable": True, "type": "numeric"},
                    ],
                    data=[
                        {
                            "Date": i[0],
                            "R value": i[1],
                        }
                        for i in [("2020-01-01", 3.2), ("2020-02-01", 2.9), ("2020-03-01", 2.5), ("2020-04-01", 0.8), ("2020-05-01", 1.1), ("2020-06-01", 2.0), ("2020-07-01", 2.1), ("2020-08-01", 2.2), ("2020-09-01", 2.3)]
                    ],
                    style_cell_conditional=[
                        {'if': {'column_id': 'Date'},
                         'width': '5px'},
                        {'if': {'column_id': 'R value'},
                         'width': '10px'},
                    ],
                    style_cell={'textAlign': 'left',
                                'fontSize': 16, 'font-family': 'Helvetica'},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },

                ),
            ]
        ),
        dbc.Button("Apply", id="submit-button-state",
                   color="primary", block=True)
    ],
    body=True,
)

# layout for the whole page
app.layout = dbc.Container(
    [
        # first, a jumbotron for the description and title
        dbc.Jumbotron(
            [
                dbc.Container(
                    [
                        html.H1("CoSim", className="display-3"),
                        html.P(
                            "Interactively simulate different Coronavirus scenarios. ",
                            className="lead",
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''

                            You can freely tune the date the first infection occured, the total population, the number of ICU
                            beds per 100k people (you can look the numbers up [here](https://en.wikipedia.org/wiki/List_of_countries_by_hospital_beds#Numbers)),
                            the probability of needing intensive care and the probability of dying under intensive care.
                            You can also change the reproduction rate R over time to simulate lockdowns, social distancing, a potential second wave, etc.

                            Read [this article](https://towardsdatascience.com/infectious-disease-modelling-part-i-understanding-sir-28d60e29fdfc) 
                            if you want to know more about the various parameters, 
                            and [this one](https://towardsdatascience.com/infectious-disease-modelling-beyond-the-basic-sir-model-216369c584c4) 
                            if you want to learn about the exact model used here.
                            '''
                                     )
                    ],
                    fluid=True,
                )
            ],
            fluid=True,
            className="jumbotron bg-white text-dark"
        ),
        # now onto the main page, i.e. the controls on the left
        # and the graphs on the right.
        dbc.Row(
            [
                # here we place the controls we just defined,
                # and tell them to use up the left 3/12ths of the page.
                dbc.Col(controls, md=3),
                # now we place the graphs on the page, taking up
                # the right 9/12ths.
                dbc.Col(
                    [
                        # the main graph that displays coronavirus over time.
                        dcc.Graph(id='main_graph'),
                        # the graph displaying the R values the user inputs over time.
                        dcc.Graph(id='r0_graph'),
                        # the next two graphs don't need as much space, so we
                        # put them next to each other in one row.
                        dbc.Row(
                            [
                                # the graph for the fatality rate over time.
                                dbc.Col(dcc.Graph(id='cfr_graph'), md=6),
                                # the graph for the daily deaths over time.
                                dbc.Col(dcc.Graph(id="deaths_graph"), md=6)

                            ]
                        ),
                    ],
                    md=9
                ),
            ],
            align="top",
        ),
    ],
    # fluid is set to true so that the page reacts nicely to different sizes etc.
    fluid=True,
)



############################################ the dash app callbacks ################################################


@app.callback(
    [dash.dependencies.Output('main_graph', 'figure'),
     dash.dependencies.Output('cfr_graph', 'figure'),
     dash.dependencies.Output('r0_graph', 'figure'),
     dash.dependencies.Output('deaths_graph', 'figure'),
     ],
     
    [dash.dependencies.Input('submit-button-state', 'n_clicks')],

    [dash.dependencies.State('initial_cases', 'value'),
     dash.dependencies.State('initial_date', 'date'),
     dash.dependencies.State('population', 'value'),
     dash.dependencies.State('icu_beds', 'value'),
     dash.dependencies.State('p_I_to_C', 'value'),
     dash.dependencies.State('p_C_to_D', 'value'),
     dash.dependencies.State('r0_table', 'data'),
     dash.dependencies.State('r0_table', 'columns')
     ]
)


def update_graph(_, initial_cases, initial_date, population, icu_beds, p_I_to_C, p_C_to_D, r0_data, r0_columns):
    
    last_initial_date, last_population, last_icu_beds, last_p_I_to_C, last_p_C_to_D = "2020-01-15", 1_000_000, 5.0, 5.0, 50.0
    if not (initial_date and population and icu_beds and p_I_to_C and p_C_to_D):
        initial_date, population, icu_beds, p_I_to_C, p_C_to_D = last_initial_date, last_population, last_icu_beds, last_p_I_to_C, last_p_C_to_D


    r0_data_x = [datapoint["Date"] for datapoint in r0_data]
    r0_data_y = [datapoint["R value"] if ((not np.isnan(datapoint["R value"])) and (datapoint["R value"] >= 0))  else 0 for datapoint in r0_data]
    f = interpolate.interp1d([0, 1, 2, 3, 4, 5, 6, 7, 8], r0_data_y, kind='linear')
    r0_x_dates = pd.date_range(start=np.datetime64("2020-01-01"), end=np.datetime64("2020-09-01"), freq="D")
    r0_y_interpolated = f(np.linspace(0, 8, num=len(r0_x_dates))).tolist()

    dates, S, E, I, C, R, D, R_0_over_time, total_CFR, daily_CFR, B = Model(initial_cases, initial_date, population, icu_beds, 3.0, 0.01, 50, 2.3, float(p_I_to_C)/100, float(p_C_to_D)/100, 0.001, r0_y_interpolated)

    return {  # return graph for compartments, graph for fatality rates, graph for reproduction rate, and graph for deaths over time
        'data': [
            {'x': dates, 'y': S.astype(int), 'type': 'line', 'name': 'susceptible'},
            {'x': dates, 'y': E.astype(int), 'type': 'line', 'name': 'exposed'},
            {'x': dates, 'y': I.astype(int), 'type': 'line', 'name': 'infected'},
            {'x': dates, 'y': C.astype(int), 'type': 'line', 'name': 'critical'},
            {'x': dates, 'y': R.astype(int), 'type': 'line', 'name': 'recovered'},
            {'x': dates, 'y': D.astype(int), 'type': 'line', 'name': 'dead'},
        ],
        'layout': {
            'title': 'Compartments over time'
        }
        }, {
        'data': [
            {'x': dates, 'y': daily_CFR, 'type': 'line',
                'name': 'daily'},
            {'x': dates, 'y': total_CFR, 'type': 'line',
                'name': 'total'}
        ],
        'layout': {
            'title': 'Fatality rate over time (%)',
            }
        }, {
        'data': [
            {'x': dates, 'y': R_0_over_time, 'type': 'line', 'name': 'susceptible'}
        ],
        'layout': {
            'title': 'Reproduction Rate R over time',
            }
        }, {
        'data': [
            {'x': dates, 'y': [0] + [D[i]-D[i-1] for i in range(1, len(dates))], 'type': 'line', 'name': 'total'},
            {'x': dates, 'y': [0] + [max(0, C[i-1]-B[i-1]) for i in range(1, len(dates))], 'type': 'line', 'name': 'due to overcapacity'}
        ],
        'layout': {
            'title': 'Deaths per day'
        }
        }






if __name__ == '__main__':
    app.run_server(debug=True)
