import dash
from dash import  dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import networkx as nx
from itertools import combinations
import pandas as pd
import ast
# Dash 앱 초기화
app = dash.Dash(__name__)

# 상위 30개 단어 추출 (빈도 기준)
from collections import Counter
from itertools import chain
# 중심성 함수
def create_filtered_network_from_phase(df, phase_name, min_cooccur=5, top_n=30):
    phase_df = df[df['phase'] == phase_name]
    keyword_lists = phase_df['extended'].apply(lambda x: list(set(x)))  # 중복 제거

    cooccurrence = Counter()
    for keywords in keyword_lists:
        for pair in combinations(sorted(keywords), 2):
            cooccurrence[pair] += 1

    edges = [(a, b, w) for (a, b), w in cooccurrence.items() if w >= min_cooccur]

    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    if G.number_of_nodes() <= top_n:
        return G  # 작을 경우 전체 반환

    # Degree 중심성 기준 상위 N개 노드만 남기기
    degree_dict = nx.degree_centrality(G)
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:top_n]
    subG = G.subgraph(top_nodes).copy()
    print(G.number_of_nodes(), subG.number_of_nodes())  
    
    return subG

def create_network_from_phase(df, phase_name, min_cooccur=5):
    phase_df = df[df['phase'] == phase_name]
    keyword_lists = phase_df['extended'].apply(lambda x: list(set(x)))  # 중복 제거

    cooccurrence = Counter()
    for keywords in keyword_lists:
        for pair in combinations(sorted(keywords), 2):
            cooccurrence[pair] += 1

    edges = [(a, b, w) for (a, b), w in cooccurrence.items() if w >= min_cooccur]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G

def get_plotly_edge_trace(G, pos):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    return go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

def get_plotly_node_trace(G, pos):
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    return go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=10, color='skyblue'),
        text=text,
        textposition="top center",
        hoverinfo='text'
    )

def compute_centrality(G, metric='degree'):
    if metric == 'degree':
        return nx.degree_centrality(G)
    elif metric == 'betweenness':
        return nx.betweenness_centrality(G)
    elif metric == 'closeness':
        return nx.closeness_centrality(G)
    elif metric == 'eigenvector':
        return nx.eigenvector_centrality(G, max_iter=500)
    elif metric == 'katz':
        return nx.katz_centrality(G, alpha=0.005, max_iter=500)
    else:
        raise ValueError("Unknown metric")
centrality_types = ['degree', 'betweenness', 'closeness', 'eigenvector', 'katz']

def get_plotly_node_trace(G, pos, centrality_dict):
    node_x = []
    node_y = []
    text = []
    size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(f"{node}<br>centrality: {centrality_dict[node]:.4f}")
        size.append(10 + 30 * centrality_dict[node])  # 크기 조정

    return go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=size, color='skyblue', sizemode='diameter', sizeref=1),
        text=text,
        textposition="top center",
        hoverinfo='text'
    )

def get_fixed_top_nodes(G, top_n=30, method='degree'):
    centrality = compute_centrality(G, method)
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_n]
    return top_nodes

def compute_centrality(G, metric='degree'):
    if metric == 'degree':
        return nx.degree_centrality(G)
    elif metric == 'betweenness':
        return nx.betweenness_centrality(G)
    elif metric == 'closeness':
        return nx.closeness_centrality(G)
    elif metric == 'eigenvector':
        return nx.eigenvector_centrality(G, max_iter=500)
    elif metric == 'katz':
        return nx.katz_centrality(G, alpha=0.005, max_iter=500)

extracted_df = pd.read_csv('extracted_df.csv')
extracted_df['extended'] = extracted_df['extended'].apply(lambda x: ast.literal_eval(x))  # 문자열을 리스트로 변환
top_words = Counter(chain.from_iterable(extracted_df['extended'])).most_common(30)
top_words = [word for word, _ in top_words]

phases = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']
centrality_types = ['degree', 'closeness', 'betweenness', 'eigenvector', 'katz']

# 미리 모든 phase의 pos와 edge_trace를 저장
fixed_graphs = {}
for phase in phases:
    G_full = create_network_from_phase(extracted_df, phase)
    subG = G_full.subgraph(top_words).copy()
    pos = nx.kamada_kawai_layout(subG)  # 고정 layout
    
    edge_trace = get_plotly_edge_trace(subG, pos)
    fixed_graphs[phase] = {"G": subG, "pos": pos, "edge": edge_trace}

frames = []
for phase in phases:
    G = fixed_graphs[phase]["G"]
    pos = fixed_graphs[phase]["pos"]
    edge_trace = fixed_graphs[phase]["edge"]  # 고정됨

    for metric in centrality_types:
        centrality = compute_centrality(G, metric)
        node_trace = get_plotly_node_trace(G, pos, centrality)

        frames.append(go.Frame(
            data=[edge_trace, node_trace],  # edge는 고정
            name=f"{phase}_{metric}",
            layout=go.Layout(title_text=f"Phase: {phase} | Centrality: {metric}")
        ))

# Dash 레이아웃 설정
app.layout = html.Div([
    html.H1("Network Visualization by Centrality", style={"text-align": "center"}),
    dcc.Graph(id='network-graph'),
    html.Div([
        html.Label("Select Centrality Type:"),
        dcc.Dropdown(
            id='centrality-dropdown',
            options=[{'label': metric, 'value': metric} for metric in centrality_types],
            value=centrality_types[0],  # 초기값 설정
            style={"width": "50%"}
        ),
        html.Label("Select Phase:"),
        dcc.Slider(
            id='phase-slider',
            min=0,
            max=len(phases) - 1,
            step=1,
            marks={i: phase for i, phase in enumerate(phases)},
            value=0,  # 초기값 설정
            included=False,  # 슬라이더의 값을 변경할 때 애니메이션 효과를 자연스럽게 적용
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={"padding": "20px"})
])

# 콜백 설정: 슬라이더와 드롭다운의 상태를 공유하여 업데이트
@app.callback(
    Output('network-graph', 'figure'),
    [Input('centrality-dropdown', 'value'),
     Input('phase-slider', 'value')]
)
def update_graph(selected_metric, selected_phase_idx):
    selected_phase = phases[selected_phase_idx]

    G = fixed_graphs[selected_phase]["G"]
    pos = fixed_graphs[selected_phase]["pos"]
    edge_trace = fixed_graphs[selected_phase]["edge"]
    centrality = compute_centrality(G, selected_metric)
    node_trace = get_plotly_node_trace(G, pos, centrality)

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Phase: {selected_phase} | Centrality: {selected_metric}",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            transition=dict(
                duration=600,  # << 애니메이션 길이 (밀리초)
                easing='cubic-in-out'  # << 부드럽게 전환
            ),
            uirevision="constant",  # << 인터랙션 유지 (줌 등)
            height=800
        )
    )
    return fig



# Dash 앱 실행
if __name__ == '__main__':
    app.run(debug=True)