import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set styling
sns.set_theme(style="ticks")
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# Pretty-name fixer
def pretty_name(raw):
    base = raw.replace('-', ' ').title()
    fixes = {
        'Butanta': 'Butantã',
        'jaguare': 'Jaguaré',
        'Jaguare': 'Jaguaré',
    }
    return fixes.get(base, base)

def plot_price_per_neighborhood(data_refined):
    ensure_plots_dir()
    # Compute averages for each neighborhood and category
    summary = (
        data_refined
        .groupby("bairro")[["total"]]
        .mean()
        .reset_index()
    )
    # Fix neighborhoods' name
    summary['bairro'] = summary['bairro'].apply(pretty_name)
    # Melt into long format for seaborn
    summary_long = summary.melt(id_vars="bairro",
                                value_vars="total",
                                var_name="Categoria",
                                value_name="Valor")
    # Order neighborhoods by total (descending)
    order = summary.sort_values("total", ascending=False)["bairro"]
    # Plot
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=summary_long,
        x="Valor",
        y="bairro",
        order=order,
        color='navy'
    )
    sns.despine(left=True, right=True)
    plt.title("Comparação de custos de aluguel por bairro em São Paulo", weight="bold")
    plt.xlabel("Valor total médio em Reais", labelpad=20)
    plt.ylabel("")
    plt.legend().remove()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(r'plots/price_per_neighborhood.png', dpi=300)
    print("✅ Price per neighbothood saved at .\plots")

def calculate_average_price(dataframe):
    # retorna valor médio das preços
    return dataframe['total'].mean()

def calculate_average_area(dataframe):
    # retorna valor médio das areas
    return dataframe['area'].mean()

def plot_rent_distribution(dataframe):
    ensure_plots_dir()
    plt.figure(figsize=(8, 8))
    avg = calculate_average_price(dataframe)
    plt.hist(dataframe['total'], bins=50, edgecolor='k')
    plt.axvline(x=avg, label=f'Média: R$ {round(avg)}', ls='--', c='tab:red')
    plt.xlabel('Preço total médio em Reais')
    plt.ylabel('Quantidade de imóveis')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(frameon=True)
    plt.savefig(r'plots/rent_distribution.png', dpi=300)
    print("✅ Distribution of total rent price saved at .\plots")

def plot_area_distribution(dataframe):
    ensure_plots_dir()
    data = dataframe.copy()
    plt.figure(figsize=(8, 8))    
    data = data[data['area']<500]
    avg = calculate_average_area(data)
    plt.hist(data['area'], bins=50, edgecolor='k')
    plt.axvline(x=avg, label=f'Média: {round(avg)} m²', ls='--', c='tab:red')
    plt.xlabel('Área total média dos imóveis em m²')
    plt.ylabel('Quantidade de imóveis')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(frameon=True)
    plt.savefig(r'plots/area_distribution.png', dpi=300)
    print("✅ Distribution of house area saved at .\plots")


def plot_correlation_matrix(dataframe):
    ensure_plots_dir()
    data = dataframe.copy()
    data = data.drop(['pet','mobiliado','metro_proximo'], axis=1)
    data = data.rename(columns={
        "aluguel": "Aluguel",
        "condominio": "Condomínio",
        "iptu": "IPTU",
        "seguro_incendio": "Seguro incêndio",
        "taxa_servico": "Taxa de serviço",
        "total": "Total",
        "area": "Área",
        "quartos": "Quartos",
        "suite": "Suíte",
        "Banheiros": "Banheiros",
        "cost_per_m2": "Custo por m²",
    })
    plt.figure(figsize=(20, 15))
    correlation_matrix = data.corr()
    ax = sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0, 
        fmt='.2f',
        vmin=-1, 
        vmax=1, 
        cbar_kws={'shrink': 0.8}
        )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Correlação entre variáveis", rotation=90, labelpad=15)
    plt.title('Matriz de correlação')
    plt.tight_layout()
    # plt.xticks(rotation=35)
    plt.savefig(r'plots/correlation_matrix.png', dpi=300)
    print("✅ Correlation matrix plotted saved at .\plots")


def print_cluster_profiles(dataframe):
    print("=" * 80)
    print("Cluster Profiles (Key Metrics):")
    print("=" * 80)
    for cluster_id in sorted(dataframe['cluster'].unique()):
        cluster_data = dataframe[dataframe['cluster'] == cluster_id]
        print(f"\nCLUSTER {cluster_id} ({len(cluster_data)} propriedades - {len(cluster_data)/len(dataframe)*100:.1f}%):")
        print(f"  • Área média: {cluster_data['area'].mean():.1f} m² (median: {cluster_data['area'].median():.1f})")
        print(f"  • Aluguel médio: R$ {cluster_data['total'].mean():.0f} (median: R$ {cluster_data['total'].median():.0f})")
        print(f"  • Preço/m² médio: R$ {cluster_data['cost_per_m2'].mean():.0f}")
        print(f"  • Média de número de quartos: {cluster_data['quartos'].mean():.1f}")
        print(f"  • Próximo ao metrô?: {cluster_data['metro_proximo'].mean()*100:.0f}%")
        print(f"  • Mobiliado?: {cluster_data['mobiliado'].mean()*100:.0f}%")
        print(f"  • Aceita pet?: {cluster_data['pet'].mean()*100:.0f}%")

def plot_clusters_price_boxplot(dataframe):
    ensure_plots_dir()
    plt.figure(figsize=(20, 20))
    dataframe.boxplot(column='total', by='cluster')
    plt.title('Distribuição de preço de aluguel por cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Aluguel total (R$)')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(r'plots/clusters_price_boxplot.png', dpi=300)
    print("✅ Clusters total price boxplot saved at .\plots")

def plot_clusters_area_boxplot(dataframe):
    ensure_plots_dir()
    plt.figure(figsize=(20, 20))
    dataframe.boxplot(column='area', by='cluster')
    plt.title('Distribuição de área total por cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Área (m²)')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(r'plots/clusters_area_boxplot.png', dpi=300)
    print("✅ Clusters total area boxplot saved at .\plots")

def plot_clusters_price_per_sm_boxplot(dataframe):
    plt.figure(figsize=(20, 20))
    dataframe.boxplot(column='cost_per_m2', by='cluster')
    plt.title('Valor total por metro quadrado por cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Aluguel total por metro quadrado (R$/m²)')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(r'plots/clusters_price_per_sm_boxplot.png', dpi=300)
    print("✅ Clusters' price per m2 for saved at .\plots")

def plot_room_dist_per_cluster(dataframe):
    ensure_plots_dir()
    plt.figure(figsize=(20, 20))
    cluster_rooms = pd.crosstab(dataframe['cluster'], dataframe['quartos'], normalize='index') * 100
    cluster_rooms.plot(kind='bar', stacked=True, colormap='viridis')
    plt.xlabel('Cluster')
    plt.ylabel('Percentual (%)')
    plt.legend(title='Quartos', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tick_params(axis='x', rotation=0)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(r'plots/room_dist_per_cluster.png', dpi=300)
    print("✅ Rooms per cluster plot saved at .\plots")

def business_insights(dataframe):
    # Business insights and recommendations
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS E RECOMENDAÇÕES")
    print("="*60)
    # Calculate cluster insights
    insights = {}
    for cluster_id in sorted(dataframe['cluster'].unique()):
        cluster_data = dataframe[dataframe['cluster'] == cluster_id]
        insights[cluster_id] = {
            'size': len(cluster_data),
            'avg_rent': cluster_data['total'].mean(),
            'avg_area': cluster_data['area'].mean(),
            'avg_price_per_sqm': cluster_data['cost_per_m2'].mean(),
            'metro_access': cluster_data['metro_proximo'].mean()
        }
    # Sort clusters by average rent for easier interpretation
    sorted_clusters = sorted(insights.keys(), key=lambda x: insights[x]['avg_rent'])
    print("\n" + "-" * 80)
    print("SEGMENTOS DE MERCADO IDENTIFICADOS:")
    print("-" * 80)
    # segments
    segment_names = ['Economy', 'Custo-benefício', 'Confort', 'Premium']
    # print segments features
    for i, cluster_id in enumerate(sorted_clusters):
        segment_name = segment_names[i] if i < len(segment_names) else f'Segment {i+1}'
        data = insights[cluster_id]
        print(f"SEGMENTO {segment_name.upper()} (Cluster {cluster_id}):")
        print(f"  Tamanho do mercado: {data['size']} properties ({data['size']/len(dataframe)*100:.1f}% of market)")
        print(f"  Aluguel médio: R$ {data['avg_rent']:.0f}")
        print(f"  Área média: {data['avg_area']:.1f} m²")
        print(f"  Aluguel/m²: R$ {data['avg_price_per_sqm']:.0f}")
        print(f"  Acesso ao Metro: {data['metro_access']*100:.0f}%")

def recommendations():
    print(f"\n{'='*60}")
    print("RECOMENDAÇÕES DE NEGÓCIOS:")
    print("-" * 20)
    print("1. ESTRATÉGIA DE PRECIFICAÇÃO: Usar modelos de clusterização para identificar faixar salariais adequadas para cada público.")
    print("2. OPORTUNIDADES DE INVESTIMENTOS: Clusters com alto aluguel/m² indicativos de mercado de alto padrão a ser explorado.")
    print("3. OTIMIZAÇÃO DE ESTRATÉGIAS DE MERCADO: propriedades próximas ao metrô são os carro-chefes de moradias da linha econômica.")

def ensure_plots_dir():
    """Cria o diretório plots se ele não existir"""
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("📁 Diretório 'plots' criado")