import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import webbrowser

df = pd.read_csv(
    #"salesfunnel.xlsx"
    'radio12.csv'
)

pv = pd.pivot_table(df, index=['Name'], columns=["Status"], values=['Quantity'], aggfunc=sum, fill_value=0)


trace1 = go.Bar(x=pv.index, y=pv[('Quantity', 'declined')], name='Declined')
trace2 = go.Bar(x=pv.index, y=pv[('Quantity', 'pending')], name='Pending')
trace3 = go.Bar(x=pv.index, y=pv[('Quantity', 'presented')], name='Presented')
trace4 = go.Bar(x=pv.index, y=pv[('Quantity', 'completed')], name='completed')

app = dash.Dash()
app.layout = html.Div(children=[
    html.H1(children='Hospital Radiology Report'),
    html.Div(children='''Doctor Radiology Data'''),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [trace1, trace2, trace3, trace4],
            'layout':
            go.Layout(title='Action by Radiologist', barmode='stack')
        })
])

webbrowser.open_new_tab('http://127.0.0.1:8050/')

if __name__ == '__main__':
    app.run_server(debug=True)